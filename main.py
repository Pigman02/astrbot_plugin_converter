import os
import sys
import shutil
import asyncio
import subprocess
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Set

# ================= 框架核心导入 =================
from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.star import Context, Star, StarTools
from astrbot.api.message_components import Image, Plain, File, Video
from astrbot.core.utils.session_waiter import session_waiter, SessionController
from astrbot.api import logger
# ===============================================

# ================= 第三方库导入 =================
try:
    import aiohttp
    from moviepy.editor import VideoFileClip, vfx
    from pypdf import PdfWriter
    from PIL import Image as PILImage
    from playwright.async_api import async_playwright, Playwright, Browser
except ImportError as e:
    logger.error(f"插件 astrbot_plugin_converter 依赖缺失: {e}")
    logger.error("请确保 requirements.txt 内容为: moviepy==1.0.3 pypdf Pillow playwright aiohttp")
    raise e
# ===============================================

class Toolbox(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        
        self.base_dir = StarTools.get_data_dir("astrbot_plugin_converter") 
        self.temp_dir = self.base_dir / "temp"
        self.rules_file = self.base_dir / "adblock_rules.txt"
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self._browser_lock = asyncio.Lock()
        
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        self.ad_domains: Set[str] = set()
        
        asyncio.create_task(self._init_adblock_rules())

    # =======================================================
    # 广告拦截核心
    # =======================================================

    async def _init_adblock_rules(self):
        if not self.config.get("screenshot_config", {}).get("enable_adblock", True):
            return
        if not self.rules_file.exists():
            logger.info("Toolbox: 本地无规则库，准备初始化...")
            await self._update_adblock_rules()
        else:
            await self._load_rules_to_memory()

    async def _update_adblock_rules(self):
        cfg = self.config.get("screenshot_config", {})
        urls = cfg.get("adblock_list_urls", [])
        if not urls:
            urls = ["https://raw.githubusercontent.com/AdAway/adaway.github.io/master/hosts.txt"]
        
        proxy = cfg.get("proxy_url", "")
        combined_content = ""
        success_count = 0

        logger.info(f"Toolbox: 开始更新 {len(urls)} 个广告规则源...")

        async with aiohttp.ClientSession() as session:
            for url in urls:
                try:
                    logger.info(f"Toolbox: 正在下载规则 -> {url}")
                    async with session.get(url, proxy=proxy if proxy else None, timeout=20) as resp:
                        if resp.status == 200:
                            text = await resp.text()
                            combined_content += text + "\n"
                            success_count += 1
                        else:
                            logger.warning(f"Toolbox: 规则源下载失败 [{resp.status}]: {url}")
                except Exception as e:
                    logger.warning(f"Toolbox: 规则源连接异常: {e} - {url}")

        if success_count > 0:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self.executor, self._write_rules_file, combined_content)
                logger.info(f"Toolbox: 规则库更新完成，共合并 {success_count} 个源。")
                await self._load_rules_to_memory()
            except Exception as e:
                logger.error(f"Toolbox: 规则文件写入失败: {e}")
        else:
            logger.error("Toolbox: 所有规则源均下载失败，请检查网络或代理设置。")

    def _write_rules_file(self, content: str):
        with open(self.rules_file, "w", encoding="utf-8") as f:
            f.write(content)

    async def _load_rules_to_memory(self):
        if not self.rules_file.exists(): return
        loop = asyncio.get_running_loop()
        self.ad_domains = await loop.run_in_executor(self.executor, self._parse_rules_file)
        logger.info(f"Toolbox: 内存已加载 {len(self.ad_domains)} 条广告屏蔽规则")

    def _parse_rules_file(self) -> Set[str]:
        temp_set = set()
        try:
            with open(self.rules_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("!"): continue
                    parts = line.split()
                    if len(parts) >= 2:
                        temp_set.add(parts[1])
                    elif len(parts) == 1 and "." in line:
                        temp_set.add(line)
        except Exception as e:
            logger.error(f"Toolbox: 规则解析异常: {e}")
        return temp_set

    async def _setup_page(self, page, event: AstrMessageEvent):
        cfg = self.config.get("screenshot_config", {})
        await page.set_viewport_size({"width": cfg.get("width", 1920), "height": cfg.get("height", 1080)})

        if cfg.get("enable_adblock", True):
            await page.add_style_tag(content="""
                div[class*="ad-"], div[id*="ad-"], div[class*="banner"], 
                iframe[src*="ads"], iframe[src*="google"], .adsbygoogle, .g-ads, 
                #google_ads_frame, [id^="google_ads_"], [id^="div-gpt-ad"] {
                    display: none !important; height: 0 !important; width: 0 !important; visibility: hidden !important;
                }
            """)
            block_types = {"image", "media", "font", "script", "xhr", "fetch", "websocket", "other"}
            custom_keywords = cfg.get("custom_block_list", [])

            async def route_handler(route):
                req = route.request
                if req.resource_type in block_types:
                    try:
                        hostname = urlparse(req.url).hostname
                        if hostname and hostname in self.ad_domains: return await route.abort()
                        url_str = req.url.lower()
                        for kw in custom_keywords:
                            if kw.replace('*', '') in url_str: return await route.abort()
                    except: pass
                await route.continue_()
            await page.route("**/*", route_handler)

    # =======================================================
    # 资源管理
    # =======================================================
    
    async def _get_browser(self) -> Browser:
        async with self._browser_lock:
            if self.browser and self.browser.is_connected(): return self.browser
            if not self.playwright: self.playwright = await async_playwright().start()
            
            shot_cfg = self.config.get("screenshot_config", {})
            proxy_url = shot_cfg.get("proxy_url", "")
            launch_args = {"headless": True, "args": ["--disable-blink-features=AutomationControlled"]}
            if proxy_url:
                logger.info(f"Toolbox: 使用代理启动浏览器 -> {proxy_url}")
                launch_args["proxy"] = {"server": proxy_url}

            self.browser = await self.playwright.firefox.launch(**launch_args)
            return self.browser

    def _install_firefox_sync(self):
        logger.info("Toolbox: 正在安装 Firefox...")
        cmd = [sys.executable, "-m", "playwright", "install", "firefox"]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Toolbox: Firefox 安装成功")

    async def _ensure_browser_env(self, event: AstrMessageEvent) -> Browser:
        try: return await self._get_browser()
        except Exception as e:
            if "executable" in str(e).lower() or "not found" in str(e).lower():
                logger.warning("Toolbox: Firefox内核缺失，触发自动安装")
                if event: await event.send(MessageChain([Plain("正在初始化 Firefox 内核，请稍候...")]))
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self.executor, self._install_firefox_sync)
                return await self._get_browser()
            else: raise e

    async def terminate(self):
        logger.info("Toolbox: 正在清理资源...")
        if self.browser:
            try: await self.browser.close()
            except: pass
        if self.playwright:
            try: await self.playwright.stop()
            except: pass
        self.executor.shutdown(wait=False)
        if self.temp_dir.exists(): shutil.rmtree(self.temp_dir, ignore_errors=True)

    # =======================================================
    # 核心功能
    # =======================================================

    async def _auto_scroll(self, page):
        if not self.config.get("screenshot_config", {}).get("enable_auto_scroll", True): return
        await page.evaluate("""
            async () => {
                await new Promise((resolve, reject) => {
                    var totalHeight = 0; var distance = 200;
                    var timer = setInterval(() => {
                        var scrollHeight = document.body.scrollHeight;
                        window.scrollBy(0, distance); totalHeight += distance;
                        if(totalHeight >= scrollHeight || totalHeight > 50000){ clearInterval(timer); resolve(); }
                    }, 80);
                });
            }
        """)
        await page.wait_for_timeout(800)

    async def _core_screenshot(self, event: AstrMessageEvent, url: str) -> str:
        if not url.startswith("http"): url = "https://" + url
        browser = await self._ensure_browser_env(event)
        page = await browser.new_page()
        try:
            await self._setup_page(page, event)
            await page.goto(url, wait_until="networkidle", timeout=60000)
            await self._auto_scroll(page)
            path = self.temp_dir / f"shot_{int(os.times().elapsed)}.png"
            await page.screenshot(path=str(path), full_page=True)
            return str(path)
        finally: await page.close()

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
                    await page.goto(url, wait_until="networkidle", timeout=90000)
                    await self._auto_scroll(page)
                    img_path = self.temp_dir / f"tmp_{idx}.png"
                    await page.screenshot(path=str(img_path), full_page=True)
                    img = PILImage.open(str(img_path))
                    if img.mode == 'RGBA': img = img.convert('RGB')
                    pdf_path = self.temp_dir / f"tmp_{idx}.pdf"
                    img.save(str(pdf_path), "PDF", resolution=100.0)
                    temp_pdfs.append(pdf_path)
                    os.remove(img_path)
                except Exception as e: logger.error(f"Page failed: {url} - {e}")
                finally: await page.close()

            if not temp_pdfs: raise Exception("所有网页处理失败")
            final_path = self.temp_dir / "WebCollection.pdf"
            merger = PdfWriter()
            for pdf in temp_pdfs: merger.append(str(pdf))
            merger.write(str(final_path))
            merger.close()
            for pdf in temp_pdfs:
                try: os.remove(pdf)
                except: pass
            return str(final_path)
        except Exception: raise

    # [核心修复] 对QQ回复消息的增强检测
    async def _process_speed(self, event: AstrMessageEvent, speed_factor: float, fps: int = 15):
        media_url = None
        is_video = False
        
        # 1. 优先检查当前消息链
        for comp in event.message_obj.message:
            if isinstance(comp, Image):
                media_url = comp.url; break
            elif isinstance(comp, Video):
                media_url = comp.url; is_video = True; break
        
        # 2. 如果没找到，检查引用回复 (OneBot 特性)
        if not media_url:
            try:
                raw = event.message_obj.raw_message
                # 检查是否存在 reply 字段
                if isinstance(raw, dict) and 'reply' in raw:
                    reply_data = raw['reply']
                    # reply['message'] 通常是 segment 列表
                    if isinstance(reply_data, dict) and 'message' in reply_data:
                        segs = reply_data['message']
                        if isinstance(segs, list):
                            for seg in segs:
                                if seg.get('type') == 'image':
                                    media_url = seg.get('data', {}).get('url')
                                    break
                                elif seg.get('type') == 'video':
                                    media_url = seg.get('data', {}).get('url')
                                    is_video = True
                                    break
            except Exception as e:
                logger.warning(f"解析回复消息失败: {e}")

        if not media_url:
            await event.send(MessageChain([Plain("请发送或回复一张 GIF 或视频。")]))
            return

        await event.send(MessageChain([Plain(f"正在处理 (倍率 {speed_factor:.1f}x)...")]))

        try:
            local_path = self.temp_dir / f"src_{int(os.times().elapsed)}"
            local_path = local_path.with_suffix(".mp4" if is_video else ".gif")
            
            async with aiohttp.ClientSession() as sess:
                async with sess.get(media_url) as resp:
                    if resp.status != 200:
                        await event.send(MessageChain([Plain("下载失败。")]))
                        return
                    with open(local_path, 'wb') as f: f.write(await resp.read())

            out_path = self.temp_dir / f"out_{int(os.times().elapsed)}.gif"
            
            def _convert_task():
                clip = VideoFileClip(str(local_path))
                new_clip = clip.fx(vfx.speedx, speed_factor)
                new_clip.write_gif(str(out_path), fps=fps)
                clip.close()
                new_clip.close()

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.executor, _convert_task)

            await event.send(MessageChain([Image.fromFileSystem(str(out_path))]))
            
            if local_path.exists(): os.remove(local_path)
            if out_path.exists(): os.remove(out_path)

        except Exception as e:
            logger.error(f"Speed Error: {e}")
            await event.send(MessageChain([Plain(f"处理失败: {e}")]))

    # =======================================================
    # 指令区
    # =======================================================

    @filter.command("updatedb")
    async def update_db_cmd(self, event: AstrMessageEvent):
        yield event.plain_result("正在下载最新广告规则...")
        await self._update_adblock_rules()
        yield event.plain_result(f"更新完成，规则数: {len(self.ad_domains)}")

    @filter.command("加速")
    async def speed_up_cmd(self, event: AstrMessageEvent, factor: str = "2", fps: str = "15"):
        try:
            f = float(factor)
            target_fps = int(fps)
            if f <= 1: 
                yield event.plain_result("倍数需大于1")
                return
            await self._process_speed(event, f, target_fps)
        except ValueError:
            yield event.plain_result("参数错误")

    @filter.command("减速")
    async def speed_down_cmd(self, event: AstrMessageEvent, factor: str = "2", fps: str = "15"):
        try:
            f = float(factor)
            target_fps = int(fps)
            if f <= 0:
                yield event.plain_result("倍数需大于0")
                return
            await self._process_speed(event, 1.0/f, target_fps)
        except ValueError:
            yield event.plain_result("参数错误")

    @filter.command("web2img")
    async def web_screenshot_cmd(self, event: AstrMessageEvent, url: str):
        try:
            path = await self._core_screenshot(event, url)
            yield event.image_result(str(path))
        except Exception as e:
            yield event.plain_result(f"截图失败: {e}")

    @filter.command("web2pdf")
    async def web_to_pdf_cmd(self, event: AstrMessageEvent, urls: str):
        yield event.plain_result("正在处理...")
        try:
            path = await self._core_web_to_pdf(event, urls)
            yield event.chain_result([File(file=str(path), name="网页合集.pdf")])
        except Exception as e:
            yield event.plain_result(f"处理失败: {e}")

    @filter.command("ocr")
    async def ocr_cmd(self, event: AstrMessageEvent):
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
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "OCR task: Extract all text."},
                        {"type": "image_url", "image_url": {"url": img_url}}
                    ]}],
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
                        await ctx.send(ctx.plain_result(f"失败: {e}"))
                    finally:
                        for f in pdf_files:
                            try: os.remove(f)
                            except: pass
                controller.stop()
                return
            
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

    # =======================================================
    # LLM Tools
    # =======================================================

    @filter.llm_tool(name="web_screenshot")
    async def web_screenshot_tool(self, event: AstrMessageEvent, url: str):
        try:
            path = await self._core_screenshot(event, url)
            await event.send(MessageChain([Image.fromFileSystem(str(path))]))
            return "截图已发送"
        except Exception as e: return f"失败: {e}"

    @filter.llm_tool(name="multi_web_to_pdf")
    async def multi_web_to_pdf_tool(self, event: AstrMessageEvent, urls: str):
        try:
            await event.send(MessageChain([Plain("正在处理(Firefox)...")]))
            path = await self._core_web_to_pdf(event, urls)
            await event.send(MessageChain([File(file=str(path), name="网页合集.pdf")]))
            return "PDF已发送"
        except Exception as e: return f"失败: {e}"

    @filter.llm_tool(name="convert_image")
    async def convert_image_tool(self, event: AstrMessageEvent, target_format: str):
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
            if target in ['jpg', 'jpeg'] and img.mode in ('RGBA', 'P'): img = img.convert('RGB')
            out = self.temp_dir / f"cvt.{target}"
            img.save(str(out))
            await event.send(MessageChain([Image.fromFileSystem(str(out))]))
            return "转换成功"
        except Exception as e: return f"错误: {e}"
