import os
import sys
import shutil
import asyncio
import base64
import time
import subprocess
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Set, Tuple

# ================= 框架核心导入 =================
from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.api.message_components import Image, Plain, File, Video, Reply
from astrbot.core.utils.session_waiter import session_waiter, SessionController
from astrbot.api import logger
# ===============================================

# ================= 第三方库导入 =================
try:
    import aiohttp
    # 注意：需安装 moviepy==1.0.3，新版 API 有变动
    from moviepy.editor import VideoFileClip, vfx 
    from pypdf import PdfWriter
    from PIL import Image as PILImage
    from playwright.async_api import async_playwright, Playwright, Browser
except ImportError as e:
    logger.error(f"插件 astrbot_plugin_converter 依赖缺失: {e}")
    logger.error("请确保 requirements.txt 内容为: moviepy==1.0.3 pypdf Pillow playwright aiohttp")
    # 可以在这里选择不抛出异常，而是让功能失效，防止整个 Bot 崩溃
    # raise e 

@register("toolbox", "YourName", "多功能工具箱(截图/PDF/OCR/GIF加速)", "1.1.1")
class Toolbox(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        
        # 数据目录
        self.base_dir = StarTools.get_data_dir("astrbot_plugin_converter") 
        self.temp_dir = self.base_dir / "temp"
        self.rules_file = self.base_dir / "adblock_rules.txt"
        
        # 确保目录存在
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 浏览器资源
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self._browser_lock = asyncio.Lock()
        
        # 线程池 (用于耗时操作)
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # 广告规则缓存
        self.ad_domains: Set[str] = set()
        
        # 异步初始化广告规则
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
            # 注入 CSS 隐藏常见广告元素
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
                # 仅拦截非主文档请求
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
    # 资源管理 (Browser)
    # =======================================================
    
    async def _get_browser(self) -> Browser:
        async with self._browser_lock:
            if self.browser and self.browser.is_connected(): return self.browser
            if not self.playwright: self.playwright = await async_playwright().start()
            
            shot_cfg = self.config.get("screenshot_config", {})
            proxy_url = shot_cfg.get("proxy_url", "")
            
            # 使用 Firefox，兼容性较好且容易去指纹
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
        try: 
            return await self._get_browser()
        except Exception as e:
            if "executable" in str(e).lower() or "not found" in str(e).lower():
                logger.warning("Toolbox: Firefox内核缺失，触发自动安装")
                if event: await event.send(MessageChain([Plain("正在初始化 Firefox 内核，请稍候...")]))
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self.executor, self._install_firefox_sync)
                return await self._get_browser()
            else: 
                raise e

    async def terminate(self):
        """插件卸载/重载时清理资源"""
        logger.info("Toolbox: 正在清理资源...")
        if self.browser:
            try: await self.browser.close()
            except: pass
        if self.playwright:
            try: await self.playwright.stop()
            except: pass
        self.executor.shutdown(wait=False)
        if self.temp_dir.exists(): 
            try: shutil.rmtree(self.temp_dir, ignore_errors=True)
            except: pass

    # =======================================================
    # 核心功能实现
    # =======================================================

    async def _auto_scroll(self, page):
        """自动滚动页面以触发懒加载"""
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
        # 创建新页面 (上下文独立)
        page = await browser.new_page()
        try:
            await self._setup_page(page, event)
            await page.goto(url, wait_until="networkidle", timeout=60000)
            await self._auto_scroll(page)
            
            path = self.temp_dir / f"shot_{int(time.time())}.png"
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
                    await page.goto(url, wait_until="networkidle", timeout=90000)
                    await self._auto_scroll(page)
                    
                    # 策略: 先截图再转PDF (保证所见即所得，避免打印样式丢失)
                    img_path = self.temp_dir / f"tmp_{idx}.png"
                    await page.screenshot(path=str(img_path), full_page=True)
                    
                    # 转换图片为PDF
                    img = PILImage.open(str(img_path))
                    if img.mode == 'RGBA': img = img.convert('RGB')
                    pdf_path = self.temp_dir / f"tmp_{idx}.pdf"
                    img.save(str(pdf_path), "PDF", resolution=100.0)
                    temp_pdfs.append(pdf_path)
                    
                    # 清理中间图片
                    try: os.remove(img_path)
                    except: pass
                    
                except Exception as e: 
                    logger.error(f"Page failed: {url} - {e}")
                finally: 
                    await page.close()

            if not temp_pdfs: raise Exception("所有网页处理失败")
            
            final_path = self.temp_dir / f"WebCollection_{int(time.time())}.pdf"
            merger = PdfWriter()
            for pdf in temp_pdfs: merger.append(str(pdf))
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
    # 媒体提取与GIF/视频加速逻辑
    # =======================================================

    def _extract_media_from_data(self, data: list | dict) -> Tuple[Optional[str], bool]:
        """
        从 OneBot 的 message 结构中提取图片/视频 URL
        返回: (url, is_video)
        """
        if isinstance(data, list):
            for seg in data:
                if not isinstance(seg, dict): continue
                seg_type = seg.get('type')
                seg_data = seg.get('data', {})
                
                # 提取逻辑: 优先 url, 其次 file(若是http)
                url = seg_data.get('url')
                if not url:
                    file_val = seg_data.get('file', '')
                    if file_val and str(file_val).startswith('http'):
                        url = file_val
                
                if url:
                    if seg_type == 'video': return url, True
                    if seg_type == 'image': return url, False
        return None, False

    async def _process_speed(self, event: AstrMessageEvent, speed_factor: float, fps: int = 15):
        media_url = None
        is_video = False
        
        # 1. 优先检查当前消息链 (例如: /加速 [图片])
        for comp in event.message_obj.message:
            if isinstance(comp, Image) and comp.url:
                media_url = comp.url
                break
            elif isinstance(comp, Video) and comp.url:
                media_url = comp.url; is_video = True
                break
        
        # 2. 如果没找到，检查引用回复
        if not media_url:
            # 修复：使用 event.raw_message 获取原始字典，而非 message_obj.raw_message (通常是字符串)
            # 兼容不同版本的 AstrBot 事件结构
            raw = getattr(event, 'raw_message', None) or getattr(event, 'raw_data', {})
            
            if isinstance(raw, dict) and 'reply' in raw:
                reply_payload = raw['reply']
                # 兼容不同适配器字段
                possible_msgs = reply_payload.get('message') or reply_payload.get('content')
                if possible_msgs:
                    media_url, is_video = self._extract_media_from_data(possible_msgs)

        if not media_url:
            await event.send(MessageChain([Plain("未找到图片或视频。请直接发送图片，或回复一张图片/视频并输入指令。")]))
            return

        await event.send(MessageChain([Plain(f"正在处理 (倍率 {speed_factor:.1f}x)...")]))

        local_path = None
        out_path = None

        try:
            # 下载文件
            suffix = ".mp4" if is_video else ".gif"
            local_path = self.temp_dir / f"src_{int(time.time())}{suffix}"
            
            async with aiohttp.ClientSession() as sess:
                async with sess.get(media_url) as resp:
                    if resp.status != 200:
                        await event.send(MessageChain([Plain("媒体文件下载失败。")]))
                        return
                    content = await resp.read()
                    with open(local_path, 'wb') as f: f.write(content)

            out_path = self.temp_dir / f"out_{int(time.time())}.gif"
            
            # 在线程池中进行耗时转换
            def _convert_task():
                clip = None
                new_clip = None
                try:
                    clip = VideoFileClip(str(local_path))
                    # MoviePy 1.x 语法
                    new_clip = clip.fx(vfx.speedx, speed_factor)
                    new_clip.write_gif(str(out_path), fps=fps, verbose=False, logger=None)
                except Exception as e:
                    logger.error(f"MoviePy Convert Error: {e}")
                    raise e
                finally:
                    # 修复: 必须显式关闭资源，否则Windows下无法删除文件
                    if new_clip: 
                        try: new_clip.close()
                        except: pass
                    if clip: 
                        try: clip.close()
                        except: pass

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.executor, _convert_task)

            if out_path.exists():
                await event.send(MessageChain([Image.fromFileSystem(str(out_path))]))
            else:
                await event.send(MessageChain([Plain("处理失败: 输出文件未生成")]))

        except Exception as e:
            logger.error(f"Speed Error: {e}")
            await event.send(MessageChain([Plain(f"处理异常: {e}")]))
        finally:
            # 清理临时文件
            if local_path and local_path.exists():
                try: os.remove(local_path)
                except: pass
            if out_path and out_path.exists():
                try: os.remove(out_path)
                except: pass

    # =======================================================
    # 指令区
    # =======================================================

    @filter.command("updatedb")
    async def update_db_cmd(self, event: AstrMessageEvent):
        """更新广告拦截规则库"""
        yield event.plain_result("正在下载最新广告规则...")
        await self._update_adblock_rules()
        yield event.plain_result(f"更新完成，规则数: {len(self.ad_domains)}")

    @filter.command("加速")
    async def speed_up_cmd(self, event: AstrMessageEvent, factor: str = "2", fps: str = "15"):
        """加速 GIF/视频: /加速 2"""
        try:
            f = float(factor)
            target_fps = int(fps)
            if f <= 1: 
                yield event.plain_result("倍数需大于1")
                return
            await self._process_speed(event, f, target_fps)
        except ValueError:
            yield event.plain_result("参数错误，请输入数字")

    @filter.command("减速")
    async def speed_down_cmd(self, event: AstrMessageEvent, factor: str = "2", fps: str = "15"):
        """减速 GIF/视频: /减速 2"""
        try:
            f = float(factor)
            target_fps = int(fps)
            if f <= 0:
                yield event.plain_result("倍数需大于0")
                return
            await self._process_speed(event, 1.0/f, target_fps)
        except ValueError:
            yield event.plain_result("参数错误，请输入数字")

    @filter.command("web2img")
    async def web_screenshot_cmd(self, event: AstrMessageEvent, url: str):
        """网页长截图: /web2img https://baidu.com"""
        try:
            yield event.plain_result("正在加载页面...")
            path = await self._core_screenshot(event, url)
            yield event.image_result(str(path))
        except Exception as e:
            yield event.plain_result(f"截图失败: {e}")

    @filter.command("web2pdf")
    async def web_to_pdf_cmd(self, event: AstrMessageEvent, urls: str):
        """网页转PDF: /web2pdf url1 url2"""
        yield event.plain_result("正在处理，耗时较长请稍候...")
        try:
            path = await self._core_web_to_pdf(event, urls)
            yield event.chain_result([File(file=str(path), name="网页合集.pdf")])
        except Exception as e:
            yield event.plain_result(f"处理失败: {e}")

    @filter.command("ocr")
    async def ocr_cmd(self, event: AstrMessageEvent):
        """OCR文字识别 (附带图片)"""
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
            yield event.plain_result("未配置 OCR API Key (config.json)")
            return

        try:
            async with aiohttp.ClientSession() as session:
                # 1. 下载图片
                async with session.get(img_url) as img_resp:
                    if img_resp.status != 200:
                        yield event.plain_result(f"图片下载失败: {img_resp.status}")
                        return
                    img_bytes = await img_resp.read()
                    base64_image = base64.b64encode(img_bytes).decode('utf-8')

                # 2. 构建请求
                # 智能判断 URL 是否包含后缀
                api_base = cfg.get("api_url", "https://api.openai.com/v1").rstrip('/')
                if "chat/completions" not in api_base:
                    api_url = f"{api_base}/chat/completions"
                else:
                    api_url = api_base

                payload = {
                    "model": cfg.get("model_name", "gpt-4o"),
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "OCR task: Extract all text from this image directly without formatting. Do not explain."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}],
                    "max_tokens": 2000
                }
                headers = {
                    "Authorization": f"Bearer {api_key}", 
                    "Content-Type": "application/json"
                }
                
                # 3. 发送请求
                async with session.post(api_url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        err_text = await resp.text()
                        yield event.plain_result(f"API错误 {resp.status}: {err_text[:100]}")
                        return
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    yield event.plain_result(content)

        except Exception as e:
            logger.error(f"OCR Error: {e}")
            yield event.plain_result(f"OCR执行出错: {e}")

    @filter.command("mergepdf")
    async def merge_pdf_cmd(self, event: AstrMessageEvent):
        """合并多个PDF文件"""
        yield event.plain_result("请发送PDF文件，发送 'end' 或 '结束' 完成合并。")
        pdf_files = []
        
        # 300秒超时
        @session_waiter(timeout=300)
        async def waiter(controller: SessionController, ctx: AstrMessageEvent):
            text = ctx.message_str.strip().lower()
            if text in ["end", "结束", "ok", "完成"]:
                if not pdf_files:
                    await ctx.send(ctx.plain_result("未收到任何文件，会话结束。"))
                else:
                    out = self.temp_dir / f"Merged_{int(time.time())}.pdf"
                    try:
                        merger = PdfWriter()
                        for f in pdf_files: merger.append(str(f))
                        merger.write(str(out))
                        merger.close()
                        await ctx.send(ctx.chain_result([File(file=str(out), name="合并结果.pdf")]))
                    except Exception as e:
                        await ctx.send(ctx.plain_result(f"合并失败: {e}"))
                    finally:
                        # 清理上传的临时文件
                        for f in pdf_files:
                            try: os.remove(f)
                            except: pass
                controller.stop()
                return
            
            # 检查文件上传
            file_url = None
            for comp in ctx.message_obj.message:
                if isinstance(comp, File) and comp.url:
                    file_url = comp.url
                    break
            
            if file_url:
                local = self.temp_dir / f"upload_{len(pdf_files)}_{int(time.time())}.pdf"
                try:
                    if file_url.startswith("http"):
                        async with aiohttp.ClientSession() as sess:
                            async with sess.get(file_url) as resp:
                                with open(local, 'wb') as f: f.write(await resp.read())
                    else:
                        shutil.copy(file_url, local)
                    
                    pdf_files.append(local)
                    await ctx.send(ctx.plain_result(f"已接收第 {len(pdf_files)} 个文件 (输入 '结束' 以合并)"))
                except Exception as e:
                    await ctx.send(ctx.plain_result(f"文件接收失败: {e}"))
            
            controller.keep() 
        
        # 启动会话等待
        await waiter(event)

    # =======================================================
    # LLM Tools (供大模型调用)
    # =======================================================

    @filter.llm_tool(name="web_screenshot")
    async def web_screenshot_tool(self, event: AstrMessageEvent, url: str):
        """截图网页"""
        try:
            path = await self._core_screenshot(event, url)
            await event.send(MessageChain([Image.fromFileSystem(str(path))]))
            return "截图已发送"
        except Exception as e: return f"失败: {e}"

    @filter.llm_tool(name="multi_web_to_pdf")
    async def multi_web_to_pdf_tool(self, event: AstrMessageEvent, urls: str):
        """将多个网页转为PDF"""
        try:
            await event.send(MessageChain([Plain("正在处理(Firefox)...")]))
            path = await self._core_web_to_pdf(event, urls)
            await event.send(MessageChain([File(file=str(path), name="网页合集.pdf")]))
            return "PDF已发送"
        except Exception as e: return f"失败: {e}"

    @filter.llm_tool(name="convert_image")
    async def convert_image_tool(self, event: AstrMessageEvent, target_format: str):
        """转换图片格式 (jpg/png/webp)"""
        target = target_format.lower().replace('.', '')
        if target not in ['jpg', 'jpeg', 'png', 'webp', 'bmp']: return "不支持该格式"
        
        img_url = None
        for comp in event.message_obj.message:
            if isinstance(comp, Image):
                img_url = comp.url
                break
        if not img_url: return "请在对话中发送一张图片"
        
        try:
            local = self.temp_dir / f"src_{int(time.time())}"
            async with aiohttp.ClientSession() as sess:
                async with sess.get(img_url) as resp:
                    with open(local, 'wb') as f: f.write(await resp.read())
            
            img = PILImage.open(str(local))
            if target in ['jpg', 'jpeg'] and img.mode in ('RGBA', 'P'): 
                img = img.convert('RGB')
                
            out = self.temp_dir / f"cvt_{int(time.time())}.{target}"
            img.save(str(out))
            
            await event.send(MessageChain([Image.fromFileSystem(str(out))]))
            return "转换成功"
        except Exception as e: return f"错误: {e}"
