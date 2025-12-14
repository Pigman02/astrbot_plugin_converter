import os
import sys
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# ================= 框架核心导入 (严格遵守审查要求) =================
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, StarTools
from astrbot.api.message_components import Image, Plain, File
from astrbot.core.utils.session_waiter import session_waiter, SessionController
from astrbot.api import logger
# ================================================================

# 第三方库导入 (带错误处理，遵循最佳实践)
try:
    import aiohttp
    from moviepy.editor import VideoFileClip
    from pypdf import PdfWriter
    from PIL import Image as PILImage
    # Playwright 异步 API
    from playwright.async_api import async_playwright, Playwright, Browser
except ImportError as e:
    logger.warning(f"Toolbox: 依赖库未完整安装，功能可能受限: {e}")

# 遵循 v3.5.20+ 建议，不再使用 @register 装饰器
class Toolbox(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        
        # [审查修正] 数据持久化: 使用 StarTools 获取规范的数据目录
        # 路径将位于: data/plugin_data/toolbox_firefox/temp
        self.base_dir = StarTools.get_data_dir("toolbox_firefox")
        self.temp_dir = self.base_dir / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Playwright 全局实例 (懒加载)
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        
        # 锁：防止并发启动/安装
        self._browser_lock = asyncio.Lock()
        
        # 线程池：专门用来跑耗时的安装命令，不卡主线程
        self.executor = ThreadPoolExecutor(max_workers=1)

    # =======================================================
    # 资源管理 (线程池异步安装 Firefox)
    # =======================================================
    
    async def _get_browser_instance_only(self) -> Browser:
        """内部方法：仅尝试获取浏览器，不处理安装逻辑"""
        async with self._browser_lock:
            if self.browser and self.browser.is_connected():
                return self.browser
            
            if not self.playwright:
                self.playwright = await async_playwright().start()
            
            # 尝试启动 Firefox
            self.browser = await self.playwright.firefox.launch(headless=True)
            return self.browser

    def _install_firefox_sync(self):
        """
        [同步方法] 在线程池中运行的安装脚本。
        使用 sys.executable 确保在当前 Python 环境中执行。
        """
        import subprocess
        logger.info("Toolbox: 开始下载 Firefox 内核...")
        try:
            cmd = [sys.executable, "-m", "playwright", "install", "firefox"]
            # check=True 会在返回码非0时抛出异常
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Toolbox: Firefox 安装成功")
        except subprocess.CalledProcessError as e:
            logger.error(f"Toolbox: Firefox 安装失败 stderr: {e.stderr}")
            raise Exception(f"安装失败: {e.stderr}")

    async def terminate(self):
        """资源清理: 关闭浏览器、停止驱动、清空临时文件"""
        logger.info("Toolbox: 正在清理资源...")
        
        # 1. 关闭 Playwright 相关
        if self.browser:
            try:
                await self.browser.close()
            except Exception: pass
            self.browser = None
            
        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception: pass
            self.playwright = None
            
        # 2. 关闭线程池
        self.executor.shutdown(wait=False)

        # 3. 清理临时文件 (操作 Path 对象)
        if self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logger.info("Toolbox: 临时缓存已清空")
            except Exception as e:
                logger.error(f"Toolbox: 临时文件清理失败: {e}")

    # =======================================================
    # 核心功能实现
    # =======================================================

    @filter.command("ocr")
    async def ocr_cmd(self, event: AstrMessageEvent):
        '''识别图片文字: /ocr (需附带图片)'''
        # 获取图片 URL
        img_url = None
        for component in event.message_obj.message:
            if isinstance(component, Image):
                img_url = component.url
                break
        
        if not img_url:
            yield event.plain_result("请在发送指令时附带图片。")
            return

        # 读取配置
        cfg = self.config.get("ocr_config", {})
        api_key = cfg.get("api_key")
        base_url = cfg.get("api_url", "").rstrip('/')
        model = cfg.get("model_name", "gpt-4o")

        if not api_key:
            yield event.plain_result("未配置 API Key。")
            return

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "OCR task: Extract all text from this image directly."},
                                {"type": "image_url", "image_url": {"url": img_url}}
                            ]
                        }
                    ],
                    "max_tokens": 2000
                }
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                url = f"{base_url}/chat/completions" if "chat/completions" not in base_url else base_url
                
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        yield event.plain_result(f"API请求失败: {resp.status}")
                        return
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    yield event.plain_result(content)
        except Exception as e:
            logger.error(f"OCR Error: {e}")
            yield event.plain_result(f"OCR Error: {e}")

    @filter.llm_tool(name="web_screenshot")
    async def web_screenshot_tool(self, event: AstrMessageEvent, url: str):
        '''网页截图。Args: url (string): 网址'''
        if not url.startswith("http"): url = "https://" + url
        
        browser = None
        try:
            # 1. 尝试获取浏览器
            try:
                browser = await self._get_browser_instance_only()
            except Exception:
                # 捕获异常，说明需要安装
                yield event.plain_result("检测到 Firefox 未安装，正在后台下载内核 (约1-2分钟)...")
                
                loop = asyncio.get_running_loop()
                # 放入线程池执行，避免阻塞主线程
                await loop.run_in_executor(self.executor, self._install_firefox_sync)
                
                yield event.plain_result("安装完成，正在执行截图任务...")
                browser = await self._get_browser_instance_only()

            # 2. 执行截图
            page = await browser.new_page()
            try:
                await page.set_viewport_size({"width": 1920, "height": 1080})
                await page.goto(url, wait_until="networkidle", timeout=30000)
                
                # 使用 Path 对象构建路径
                path = self.temp_dir / f"shot_{int(os.times().elapsed)}.png"
                await page.screenshot(path=str(path), full_page=True)
                
                yield event.image_result(str(path))
                return "截图已发送"
            finally:
                await page.close()
                
        except Exception as e:
            logger.error(f"Screenshot Error: {e}")
            return f"执行失败: {e}"

    @filter.llm_tool(name="multi_web_to_pdf")
    async def multi_web_to_pdf_tool(self, event: AstrMessageEvent, urls: str):
        '''将网页转为长图并合并为PDF。Args: urls (string): 网址，空格分隔'''
        url_list = [u.strip() for u in urls.replace(',', ' ').split(' ') if u.strip()]
        if not url_list: return "无有效URL"
        if len(url_list) > 5: return "一次最多支持5个网页"

        yield event.plain_result(f"正在处理 {len(url_list)} 个网页 (Firefox引擎)...")
        
        # 确保浏览器可用 (复用上面的逻辑)
        browser = None
        try:
            try:
                browser = await self._get_browser_instance_only()
            except Exception:
                yield event.plain_result("正在初始化 Firefox 环境...")
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self.executor, self._install_firefox_sync)
                browser = await self._get_browser_instance_only()
        except Exception as e:
            return f"环境初始化失败: {e}"

        temp_pdfs = []
        temp_imgs = []

        try:
            for idx, raw_url in enumerate(url_list):
                url = raw_url if raw_url.startswith("http") else "https://" + raw_url
                page = await browser.new_page()
                try:
                    await page.set_viewport_size({"width": 1920, "height": 1080})
                    await page.goto(url, wait_until="networkidle", timeout=45000)
                    
                    # 路径处理
                    img_path = self.temp_dir / f"tmp_{idx}.png"
                    await page.screenshot(path=str(img_path), full_page=True)
                    temp_imgs.append(img_path)
                    
                    # 图片转 PDF
                    img = PILImage.open(str(img_path))
                    if img.mode == 'RGBA': img = img.convert('RGB')
                    
                    pdf_path = self.temp_dir / f"tmp_{idx}.pdf"
                    img.save(str(pdf_path), "PDF", resolution=100.0)
                    temp_pdfs.append(pdf_path)
                    
                except Exception as e:
                    logger.error(f"Page {url} failed: {e}")
                    yield event.plain_result(f"跳过失败网页: {url}")
                finally:
                    await page.close()

            if not temp_pdfs: return "所有网页处理失败"

            # 合并 PDF
            final_path = self.temp_dir / "WebCollection.pdf"
            merger = PdfWriter()
            for pdf in temp_pdfs:
                merger.append(str(pdf))
            merger.write(str(final_path))
            merger.close()

            yield event.chain_result([File(file=str(final_path), name="网页合集.pdf")])
            
            # 立即清理
            for f in temp_imgs + temp_pdfs:
                try: os.remove(f)
                except: pass
                
            return "PDF已发送"

        except Exception as e:
            logger.error(f"Multi-PDF Error: {e}")
            return f"任务出错: {e}"

    @filter.command("mergepdf")
    async def merge_pdf_cmd(self, event: AstrMessageEvent):
        '''合并PDF: 发送指令后按提示上传多个文件'''
        yield event.plain_result("请发送PDF文件(支持多个)，发送 'end' 开始合并。")
        pdf_files = []

        @session_waiter(timeout=300)
        async def waiter(controller: SessionController, ctx: AstrMessageEvent):
            text = ctx.message_str.strip().lower()
            
            # 结束信号
            if text in ["end", "结束", "ok"]:
                if not pdf_files:
                    await ctx.send(ctx.plain_result("无文件，取消。"))
                else:
                    out_path = self.temp_dir / "Merged.pdf"
                    try:
                        merger = PdfWriter()
                        for f in pdf_files: merger.append(str(f))
                        merger.write(str(out_path))
                        merger.close()
                        await ctx.send(ctx.chain_result([File(file=str(out_path), name="合并文件.pdf")]))
                    except Exception as e:
                        await ctx.send(ctx.plain_result(f"合并失败: {e}"))
                    finally:
                        for f in pdf_files:
                            try: os.remove(f)
                            except: pass
                controller.stop()
                return

            # 接收文件
            file_url = None
            file_name = f"upload_{len(pdf_files)}.pdf"
            for comp in ctx.message_obj.message:
                if isinstance(comp, File) and comp.url:
                    file_url = comp.url
                    if comp.name: file_name = comp.name
                    break
            
            if file_url:
                local_path = self.temp_dir / file_name
                try:
                    if file_url.startswith("http"):
                        async with aiohttp.ClientSession() as sess:
                            async with sess.get(file_url) as resp:
                                with open(local_path, 'wb') as f:
                                    f.write(await resp.read())
                    else:
                        shutil.copy(file_url, local_path)
                    
                    pdf_files.append(local_path)
                    await ctx.send(ctx.plain_result(f"已接收 {len(pdf_files)} 个文件..."))
                except Exception as e:
                    await ctx.send(ctx.plain_result(f"文件接收失败: {e}"))
            
            controller.keep()

    @filter.llm_tool(name="convert_image")
    async def convert_image_tool(self, event: AstrMessageEvent, target_format: str):
        '''图片转格式。Args: target_format: jpg/png/webp'''
        target = target_format.lower().replace('.', '')
        if target not in ['jpg', 'jpeg', 'png', 'webp', 'bmp']: return "不支持的格式"
        
        img_url = None
        for comp in event.message_obj.message:
            if isinstance(comp, Image):
                img_url = comp.url
                break
        
        if not img_url: return "请先发送图片"

        try:
            local_path = self.temp_dir / f"src_{int(os.times().elapsed)}"
            async with aiohttp.ClientSession() as sess:
                async with sess.get(img_url) as resp:
                    with open(local_path, 'wb') as f: f.write(await resp.read())
            
            img = PILImage.open(str(local_path))
            if target in ['jpg', 'jpeg'] and img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            out = self.temp_dir / f"cvt.{target}"
            img.save(str(out))
            yield event.image_result(str(out))
            return "转换成功"
        except Exception as e:
            return f"转换错误: {e}"
