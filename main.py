import os
import sys
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# ================= 框架核心导入 (已修复导入路径) =================
from astrbot.api.event import filter, AstrMessageEvent
# [修复 1] MessageChain 从 astrbot.api.event 导入，而不是 message_components
from astrbot.api.event import MessageChain 
from astrbot.api.star import Context, Star, StarTools
from astrbot.api.message_components import Image, Plain, File
from astrbot.core.utils.session_waiter import session_waiter, SessionController
from astrbot.api import logger
# ================================================================

# 第三方库导入
try:
    import aiohttp
    from moviepy.editor import VideoFileClip
    from pypdf import PdfWriter
    from PIL import Image as PILImage
    from playwright.async_api import async_playwright, Playwright, Browser
except ImportError as e:
    logger.warning(f"Toolbox: 依赖库未完整安装: {e}")

class Toolbox(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        
        # 数据持久化
        self.base_dir = StarTools.get_data_dir("astrbot_plugin_converter")
        self.temp_dir = self.base_dir / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Playwright 全局实例
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self._browser_lock = asyncio.Lock()
        
        self.executor = ThreadPoolExecutor(max_workers=1)

    # =======================================================
    # 资源管理
    # =======================================================
    
    async def _get_browser_instance_only(self) -> Browser:
        async with self._browser_lock:
            if self.browser and self.browser.is_connected():
                return self.browser
            if not self.playwright:
                self.playwright = await async_playwright().start()
            self.browser = await self.playwright.firefox.launch(headless=True)
            return self.browser

    def _install_firefox_sync(self):
        import subprocess
        logger.info("Toolbox: 开始下载 Firefox 内核...")
        try:
            cmd = [sys.executable, "-m", "playwright", "install", "firefox"]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Toolbox: Firefox 安装成功")
        except subprocess.CalledProcessError as e:
            logger.error(f"Toolbox: Firefox 安装失败 stderr: {e.stderr}")
            raise Exception(f"安装失败: {e.stderr}")

    async def _ensure_browser_env(self, event: AstrMessageEvent) -> Browser:
        """确保浏览器可用，不可用则安装。使用 event.send 代替 yield"""
        try:
            return await self._get_browser_instance_only()
        except Exception:
            logger.warning("Toolbox: Firefox 未安装，触发自动安装流程。")
            if event:
                # [修复 2] 使用 MessageChain 构造消息并使用 send 发送
                await event.send(MessageChain([Plain("检测到 Firefox 未安装，正在后台下载内核 (约1-2分钟)...")]))
            
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.executor, self._install_firefox_sync)
            
            if event:
                await event.send(MessageChain([Plain("安装完成，正在执行任务...")]))
            
            return await self._get_browser_instance_only()

    async def terminate(self):
        if self.browser:
            try: await self.browser.close()
            except: pass
            self.browser = None
        if self.playwright:
            try: await self.playwright.stop()
            except: pass
            self.playwright = None
        self.executor.shutdown(wait=False)
        if self.temp_dir.exists():
            try: shutil.rmtree(self.temp_dir)
            except: pass

    # =======================================================
    # 功能实现 (工具方法中不再使用 yield，解决 SyntaxError)
    # =======================================================

    @filter.command("ocr")
    async def ocr_cmd(self, event: AstrMessageEvent):
        '''识别图片文字: /ocr (需附带图片)'''
        # 指令(Command)可以使用 yield，因为它不需要向 LLM 返回值
        img_url = None
        for component in event.message_obj.message:
            if isinstance(component, Image):
                img_url = component.url
                break
        
        if not img_url:
            yield event.plain_result("请在发送指令时附带图片。")
            return

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
                        {"role": "user", "content": [
                            {"type": "text", "text": "OCR task: Extract all text from this image directly."},
                            {"type": "image_url", "image_url": {"url": img_url}}
                        ]}
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
        
        try:
            # 1. 获取浏览器
            browser = await self._ensure_browser_env(event)

            # 2. 截图
            page = await browser.new_page()
            try:
                await page.set_viewport_size({"width": 1920, "height": 1080})
                await page.goto(url, wait_until="networkidle", timeout=30000)
                
                path = self.temp_dir / f"shot_{int(os.times().elapsed)}.png"
                await page.screenshot(path=str(path), full_page=True)
                
                # [修复 2] 使用 event.send 发送图片，不使用 yield
                await event.send(MessageChain([Image.fromFileSystem(str(path))]))
                
                # [修复 3] 返回字符串给 LLM
                return "截图已发送给用户。"
            finally:
                await page.close()
                
        except Exception as e:
            logger.error(f"Screenshot Error: {e}")
            return f"执行失败: {e}"

    @filter.llm_tool(name="multi_web_to_pdf")
    async def multi_web_to_pdf_tool(self, event: AstrMessageEvent, urls: str):
        '''将网页转为长图并合并为PDF。Args: urls (string): 网址'''
        url_list = [u.strip() for u in urls.replace(',', ' ').split(' ') if u.strip()]
        if not url_list: return "无有效URL"
        if len(url_list) > 5: return "一次最多支持5个网页"

        # 使用 send 而不是 yield
        await event.send(MessageChain([Plain(f"正在处理 {len(url_list)} 个网页 (Firefox引擎)...")]))
        
        try:
            browser = await self._ensure_browser_env(event)
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
                    
                    img_path = self.temp_dir / f"tmp_{idx}.png"
                    await page.screenshot(path=str(img_path), full_page=True)
                    temp_imgs.append(img_path)
                    
                    img = PILImage.open(str(img_path))
                    if img.mode == 'RGBA': img = img.convert('RGB')
                    
                    pdf_path = self.temp_dir / f"tmp_{idx}.pdf"
                    img.save(str(pdf_path), "PDF", resolution=100.0)
                    temp_pdfs.append(pdf_path)
                except Exception as e:
                    logger.error(f"Page {url} failed: {e}")
                    await event.send(MessageChain([Plain(f"警告: {url} 处理失败，已跳过")]))
                finally:
                    await page.close()

            if not temp_pdfs: return "所有网页处理失败"

            final_path = self.temp_dir / "WebCollection.pdf"
            merger = PdfWriter()
            for pdf in temp_pdfs:
                merger.append(str(pdf))
            merger.write(str(final_path))
            merger.close()

            # 发送文件
            await event.send(MessageChain([File(file=str(final_path), name="网页合集.pdf")]))
            
            for f in temp_imgs + temp_pdfs:
                try: os.remove(f)
                except: pass
                
            return "PDF文件已发送。"

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
            
            # 使用 send 代替 yield
            await event.send(MessageChain([Image.fromFileSystem(str(out))]))
            return f"已转换为 {target}"
        except Exception as e:
            return f"转换错误: {e}"
