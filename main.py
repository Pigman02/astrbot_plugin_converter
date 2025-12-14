import os
import sys
import shutil
import asyncio
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List

# ================= 框架核心导入 =================
from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.star import Context, Star, StarTools
from astrbot.api.message_components import Image, Plain, File
from astrbot.core.utils.session_waiter import session_waiter, SessionController
from astrbot.api import logger
# ===============================================

# 直接硬性导入，缺库直接报错，不再掩盖问题
import aiohttp
from moviepy.editor import VideoFileClip
from pypdf import PdfWriter
from PIL import Image as PILImage
from playwright.async_api import async_playwright, Playwright, Browser

class Toolbox(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        
        # 数据路径
        self.base_dir = StarTools.get_data_dir("astrbot_plugin_converter") 
        self.temp_dir = self.base_dir / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 全局实例
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self._browser_lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)

    # =======================================================
    # 资源管理 (Playwright / Firefox)
    # =======================================================
    
    async def _get_browser(self) -> Browser:
        """获取浏览器实例，如果未启动则启动"""
        async with self._browser_lock:
            if self.browser and self.browser.is_connected():
                return self.browser
            
            if not self.playwright:
                self.playwright = await async_playwright().start()
            
            # 启动 Firefox
            self.browser = await self.playwright.firefox.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"]
            )
            return self.browser

    def _install_firefox_sync(self):
        """同步安装脚本 (线程池运行)"""
        logger.info("Toolbox: 正在安装 Firefox...")
        cmd = [sys.executable, "-m", "playwright", "install", "firefox"]
        # check=True 确保失败时抛出异常
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Toolbox: Firefox 安装成功")

    async def _ensure_browser_env(self, event: AstrMessageEvent) -> Browser:
        """确保环境可用，自动处理内核缺失"""
        try:
            return await self._get_browser()
        except Exception as e:
            # 仅捕获内核缺失错误
            if "executable" in str(e).lower() or "not found" in str(e).lower():
                logger.warning("Toolbox: Firefox内核缺失，触发自动安装")
                if event:
                    await event.send(MessageChain([Plain("正在初始化 Firefox 内核，请稍候...")]))
                
                # 放入线程池安装，不阻塞主线程
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self.executor, self._install_firefox_sync)
                
                return await self._get_browser()
            else:
                raise e

    async def terminate(self):
        """清理资源"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        self.executor.shutdown(wait=False)
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # =======================================================
    # 核心业务逻辑
    # =======================================================

    async def _setup_page(self, page, event: AstrMessageEvent):
        """配置页面参数与拦截"""
        cfg = self.config.get("screenshot_config", {})
        width = cfg.get("width", 1920)
        height = cfg.get("height", 1080)
        
        await page.set_viewport_size({"width": width, "height": height})

        # 广告拦截
        if cfg.get("enable_adblock", True):
            patterns = [
                "*googleads*", "*doubleclick*", "*baidu.com/pos*", "*cnzz.com*",
                "*union.baidu*", "*adservice*", "*adsystem*", "*/ads/*"
            ]
            patterns.extend(cfg.get("custom_block_list", []))
            
            for p in patterns:
                await page.route(p, lambda route: route.abort())

    async def _core_screenshot(self, event: AstrMessageEvent, url: str) -> str:
        if not url.startswith("http"): url = "https://" + url
        browser = await self._ensure_browser_env(event)
        page = await browser.new_page()
        try:
            await self._setup_page(page, event)
            await page.goto(url, wait_until="networkidle", timeout=30000)
            
            path = self.temp_dir / f"shot_{int(os.times().elapsed)}.png"
            await page.screenshot(path=str(path), full_page=True)
            return str(path)
        finally:
            await page.close()

    async def _core_web_to_pdf(self, event: AstrMessageEvent, urls: str) -> str:
        url_list = [u.strip() for u in urls.replace(',', ' ').split(' ') if u.strip()]
        if not url_list: raise ValueError("无有效URL")

        browser = await self._ensure_browser_env(event)
        temp_pdfs = []
        
        try:
            for idx, raw_url in enumerate(url_list):
                url = raw_url if raw_url.startswith("http") else "https://" + raw_url
                page = await browser.new_page()
                try:
                    await self._setup_page(page, event)
                    await page.goto(url, wait_until="networkidle", timeout=60000)
                    
                    # 截图 -> 转 PDF (为了完美还原样式)
                    img_path = self.temp_dir / f"tmp_{idx}.png"
                    await page.screenshot(path=str(img_path), full_page=True)
                    
                    img = PILImage.open(str(img_path))
                    if img.mode == 'RGBA': img = img.convert('RGB')
                    
                    pdf_path = self.temp_dir / f"tmp_{idx}.pdf"
                    img.save(str(pdf_path), "PDF", resolution=100.0)
                    temp_pdfs.append(pdf_path)
                    
                    # 及时清理大图，释放内存
                    os.remove(img_path)
                except Exception as e:
                    logger.error(f"Page failed: {url} - {e}")
                finally:
                    await page.close()

            if not temp_pdfs: raise Exception("所有网页处理失败")

            final_path = self.temp_dir / "WebCollection.pdf"
            merger = PdfWriter()
            for pdf in temp_pdfs:
                merger.append(str(pdf))
            merger.write(str(final_path))
            merger.close()
            
            # 清理中间PDF
            for pdf in temp_pdfs:
                try: os.remove(pdf)
                except: pass
                
            return str(final_path)
        except Exception:
            raise

    # =======================================================
    # 指令与工具
    # =======================================================

    @filter.command("web2img")
    async def web_screenshot_cmd(self, event: AstrMessageEvent, url: str):
        '''网页截图: /web2img [url]'''
        # 移除了 "正在截图" 的废话回复，直接发图
        try:
            path = await self._core_screenshot(event, url)
            yield event.image_result(str(path))
        except Exception as e:
            yield event.plain_result(f"截图失败: {e}")

    @filter.command("web2pdf")
    async def web_to_pdf_cmd(self, event: AstrMessageEvent, urls: str):
        '''网页转PDF: /web2pdf [url]'''
        # PDF处理较慢，保留这个提示比较好，否则用户以为没反应
        yield event.plain_result("正在处理...")
        try:
            path = await self._core_web_to_pdf(event, urls)
            yield event.chain_result([File(file=str(path), name="网页合集.pdf")])
        except Exception as e:
            yield event.plain_result(f"处理失败: {e}")

    @filter.llm_tool(name="web_screenshot")
    async def web_screenshot_tool(self, event: AstrMessageEvent, url: str):
        '''网页截图。Args: url (string): 网址'''
        try:
            path = await self._core_screenshot(event, url)
            await event.send(MessageChain([Image.fromFileSystem(str(path))]))
            return "截图已发送"
        except Exception as e:
            return f"执行失败: {e}"

    @filter.llm_tool(name="multi_web_to_pdf")
    async def multi_web_to_pdf_tool(self, event: AstrMessageEvent, urls: str):
        '''网页转PDF。Args: urls (string): 网址'''
        try:
            await event.send(MessageChain([Plain("正在处理(Firefox)...")]))
            path = await self._core_web_to_pdf(event, urls)
            await event.send(MessageChain([File(file=str(path), name="网页合集.pdf")]))
            return "PDF已发送"
        except Exception as e:
            return f"执行失败: {e}"

    @filter.command("ocr")
    async def ocr_cmd(self, event: AstrMessageEvent):
        '''识别图片: /ocr (需附带图片)'''
        img_url = None
        for comp in event.message_obj.message:
            if isinstance(comp, Image):
                img_url = comp.url
                break
        
        if not img_url:
            yield event.plain_result("请附带图片")
            return

        cfg = self.config.get("ocr_config", {})
        api_key = cfg.get("api_key")
        if not api_key:
            yield event.plain_result("未配置API Key")
            return

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": cfg.get("model_name", "gpt-4o"),
                    "messages": [
                        {"role": "user", "content": [
                            {"type": "text", "text": "OCR task: Extract all text."},
                            {"type": "image_url", "image_url": {"url": img_url}}
                        ]}
                    ],
                    "max_tokens": 2000
                }
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                url = cfg.get("api_url", "https://api.openai.com/v1").rstrip('/')
                url = f"{url}/chat/completions" if "chat/completions" not in url else url
                
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        yield event.plain_result(f"API错误: {resp.status}")
                        return
                    data = await resp.json()
                    yield event.plain_result(data["choices"][0]["message"]["content"])
        except Exception as e:
            yield event.plain_result(f"错误: {e}")

    @filter.command("mergepdf")
    async def merge_pdf_cmd(self, event: AstrMessageEvent):
        '''合并PDF'''
        yield event.plain_result("请发送PDF，发 'end' 结束")
        pdf_files = []

        @session_waiter(timeout=300)
        async def waiter(controller: SessionController, ctx: AstrMessageEvent):
            text = ctx.message_str.strip().lower()
            if text in ["end", "结束", "ok"]:
                if not pdf_files:
                    await ctx.send(ctx.plain_result("无文件"))
                else:
                    out = self.temp_dir / "Merged.pdf"
                    try:
                        merger = PdfWriter()
                        for f in pdf_files: merger.append(str(f))
                        merger.write(str(out))
                        merger.close()
                        await ctx.send(ctx.chain_result([File(file=str(out), name="合并.pdf")]))
                    except Exception as e:
                        await ctx.send(ctx.plain_result(f"合并失败: {e}"))
                    finally:
                        for f in pdf_files:
                            try: os.remove(f)
                            except: pass
                controller.stop()
                return

            # 文件接收
            file_url = None
            for comp in ctx.message_obj.message:
                if isinstance(comp, File) and comp.url:
                    file_url = comp.url
                    break
            
            if file_url:
                local = self.temp_dir / f"upload_{len(pdf_files)}.pdf"
                try:
                    if file_url.startswith("http"):
                        async with aiohttp.ClientSession() as sess:
                            async with sess.get(file_url) as resp:
                                with open(local, 'wb') as f: f.write(await resp.read())
                    else:
                        shutil.copy(file_url, local)
                    pdf_files.append(local)
                    await ctx.send(ctx.plain_result(f"已收 {len(pdf_files)} 个"))
                except: pass
            controller.keep()

    @filter.llm_tool(name="convert_image")
    async def convert_image_tool(self, event: AstrMessageEvent, target_format: str):
        '''图片转格式'''
        target = target_format.lower().replace('.', '')
        if target not in ['jpg', 'jpeg', 'png', 'webp', 'bmp']: return "不支持"
        
        img_url = None
        for comp in event.message_obj.message:
            if isinstance(comp, Image):
                img_url = comp.url
                break
        
        if not img_url: return "请发图"

        try:
            local = self.temp_dir / f"src_{int(os.times().elapsed)}"
            async with aiohttp.ClientSession() as sess:
                async with sess.get(img_url) as resp:
                    with open(local, 'wb') as f: f.write(await resp.read())
            
            img = PILImage.open(str(local))
            if target in ['jpg', 'jpeg'] and img.mode == 'P': img = img.convert('RGB')
            if target in ['jpg', 'jpeg'] and img.mode == 'RGBA': img = img.convert('RGB')
            
            out = self.temp_dir / f"cvt.{target}"
            img.save(str(out))
            await event.send(MessageChain([Image.fromFileSystem(str(out))]))
            return "转换成功"
        except Exception as e:
            return f"错误: {e}"
