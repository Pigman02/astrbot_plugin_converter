import os
import sys
import shutil
import asyncio
import base64
import time
import re
import io
import tempfile
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Set, Tuple, Union

# ================= 框架核心导入 =================
from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.api.message_components import Image, Plain, File, Video, Reply, Node, Nodes
from astrbot.api import logger
# ===============================================

# ================= 第三方库导入 =================
try:
    import aiohttp
    import imageio
    from pypdf import PdfWriter
    from PIL import Image as PILImage, ImageSequence
    from playwright.async_api import async_playwright, Playwright, Browser
except ImportError as e:
    logger.error(f"插件依赖缺失: {e}")
    logger.error("请确保安装: pip install imageio[ffmpeg] pypdf Pillow playwright aiohttp")

@register("toolbox", "YourName", "全能工具箱(截图/PDF/OCR/GIF)", "1.3.0")
class Toolbox(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        
        # 目录配置
        self.base_dir = StarTools.get_data_dir("astrbot_plugin_toolbox") 
        self.temp_dir = self.base_dir / "temp"
        self.rules_file = self.base_dir / "adblock_rules.txt"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 资源管理
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self._browser_lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 广告拦截
        self.ad_domains: Set[str] = set()
        asyncio.create_task(self._init_adblock_rules())

    # =======================================================
    # 1. 核心工具函数 (媒体提取/文件保存)
    # =======================================================

    def _save_animation(self, output: io.BytesIO, frames: list, duration_ms: int, loop: int = 0):
        """统一保存动画，支持 GIF/APNG/WEBP"""
        fmt = self.config.get('output_format', 'GIF').upper()
        try:
            if fmt == 'APNG':
                frames[0].save(output, format='PNG', save_all=True, append_images=frames[1:], 
                             duration=duration_ms, loop=loop, optimize=True, default_image=True)
            elif fmt == 'WEBP':
                frames[0].save(output, format='WEBP', save_all=True, append_images=frames[1:], 
                             duration=duration_ms, loop=loop, method=3, quality=80)
            else:
                # GIF 默认配置
                frames[0].save(output, format='GIF', save_all=True, append_images=frames[1:], 
                             duration=duration_ms, loop=loop, optimize=True, disposal=2)
        except Exception as e:
            logger.error(f"Save animation failed, fallback to GIF. Error: {e}")
            frames[0].save(output, format='GIF', save_all=True, append_images=frames[1:], 
                         duration=duration_ms, loop=loop, optimize=True, disposal=2)

    async def _resolve_file_via_api(self, event: AstrMessageEvent, file_id: str) -> str:
        """调用 Bot API 解析 file_id"""
        try:
            res = await event.bot.api.call_action("get_file", file_id=file_id)
            if not res or not isinstance(res, dict): return None
            
            url = res.get('url')
            if url and url.startswith('http'): return url
            path = res.get('file')
            if path and os.path.exists(path): return path
            return url or path
        except: return None

    def _get_media_source(self, event: AstrMessageEvent, media_type: str = 'video') -> Optional[str]:
        """强大的媒体提取器，支持引用回复、原始数据包解析"""
        candidates = [] # (score, url/path)
        
        def extract(item):
            # 1. URL
            url = getattr(item, 'url', None)
            if not url and isinstance(item, dict):
                url = item.get('data', {}).get('url') or item.get('url')
            if url and isinstance(url, str) and url.startswith('http'):
                return 100, url
            # 2. Path
            path = getattr(item, 'path', None)
            if not path and isinstance(item, dict):
                path = item.get('data', {}).get('path') or item.get('path')
            if path and isinstance(path, str) and os.path.isabs(path) and os.path.exists(path):
                return 90, path
            # 3. File ID
            fid = getattr(item, 'file', None)
            if not fid and isinstance(item, dict):
                fid = item.get('data', {}).get('file') or item.get('file')
            if fid and isinstance(fid, str):
                return 50, fid
            return 0, None

        items = []
        # 检查 AstrBot 封装的方法
        if media_type == 'video' and hasattr(event, "get_videos"): items.extend(event.get_videos() or [])
        if media_type == 'image' and hasattr(event, "get_images"): items.extend(event.get_images() or [])
        
        # 检查原始数据 (OneBot 协议)
        raw = getattr(event, 'raw_message', None) or getattr(event, 'raw_data', {})
        if isinstance(raw, dict) and 'reply' in raw:
            reply_pl = raw['reply']
            msgs = reply_pl.get('message') or reply_pl.get('content')
            if isinstance(msgs, list): items.extend(msgs)

        # 检查当前消息链
        if hasattr(event.message_obj, "message"):
            for seg in event.message_obj.message:
                if isinstance(seg, (Image, Video, dict)):
                    # 简单类型过滤
                    if isinstance(seg, dict):
                        if seg.get('type') == media_type: items.append(seg)
                    elif (media_type == 'image' and isinstance(seg, Image)) or \
                         (media_type == 'video' and isinstance(seg, Video)):
                        items.append(seg)

        for item in items:
            s, v = extract(item)
            if v: candidates.append((s, v))
            
        if not candidates: return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    async def terminate(self):
        """资源清理"""
        logger.info("Toolbox: 清理资源...")
        if self.browser:
            try: await self.browser.close()
            except: pass
        if self.playwright:
            try: await self.playwright.stop()
            except: pass
        self.executor.shutdown(wait=False)
        try: shutil.rmtree(self.temp_dir, ignore_errors=True)
        except: pass

    # =======================================================
    # 2. 视频转GIF / 变速 / 压缩 (ImageIO实现)
    # =======================================================

    def _parse_video_args(self, text: str):
        """参数解析: 0s-5s, fps=15, 0.5x, scale=0.8"""
        params = {
            'start': 0.0, 'end': None,
            'fps': self.config.get('default_fps', 12),
            'scale': self.config.get('default_scale', 0.5), 
            'speed': 1.0 
        }
        # 时间
        time_range = re.search(r'(\d+(?:\.\d+)?)[sS]?\s*[-~]\s*(\d+(?:\.\d+)?)[sS]?', text)
        if time_range:
            params['start'] = float(time_range.group(1))
            params['end'] = float(time_range.group(2))
        # 帧率
        fps_match = re.search(r'(?:fps|帧率)[ :=]?(\d+)', text, re.IGNORECASE)
        if fps_match: params['fps'] = int(fps_match.group(1))
        # 缩放
        scale_match = re.search(r'(?:scale|缩放|大小)[ :=]?(0\.\d+|1\.0)', text)
        if scale_match: params['scale'] = float(scale_match.group(1))
        # 速度
        speed_match = re.search(r'(\d+(?:\.\d+)?)[xX]?(?:倍速|倍|speed)', text)
        if speed_match: params['speed'] = float(speed_match.group(1))
        return params

    def _process_video_core(self, video_path: str, params: dict, max_colors: int = 256):
        """转换核心逻辑"""
        try:
            reader = imageio.get_reader(video_path, format='FFMPEG')
            meta = reader.get_meta_data()
            src_fps = meta.get('fps', 30) or 30
            duration = meta.get('duration', 100)

            start_t = params['start']
            end_t = params['end'] if params['end'] else duration
            
            # 限制时长防止内存溢出
            max_dur = self.config.get('max_gif_duration', 15.0)
            if (end_t - start_t) > max_dur: end_t = start_t + max_dur

            # 计算采样步长
            target_fps = params['fps']
            base_step = max(1, src_fps / target_fps)
            final_step = max(1, int(base_step * params['speed']))
            
            frames = []
            output_fmt = self.config.get('output_format', 'GIF').upper()

            for i, frame in enumerate(reader):
                curr_t = i / src_fps
                if curr_t < start_t: continue
                if curr_t > end_t: break
                
                if i % final_step == 0:
                    pil_img = PILImage.fromarray(frame)
                    # 缩放
                    if params['scale'] != 1.0:
                        w, h = pil_img.size
                        pil_img = pil_img.resize((int(w*params['scale']), int(h*params['scale'])), PILImage.Resampling.BILINEAR)
                    # 量化
                    if output_fmt == 'GIF' and max_colors < 256:
                        pil_img = pil_img.quantize(colors=max_colors, method=1)
                    frames.append(pil_img)
                
                if len(frames) > 500: break # 安全熔断

            reader.close()
            if not frames: return None, "无有效帧", 0
            
            output = io.BytesIO()
            duration_ms = int(1000 / (src_fps / final_step))
            self._save_animation(output, frames, duration_ms)
            output.seek(0)
            size_mb = output.getbuffer().nbytes / 1024 / 1024
            return output, f"FPS:{src_fps/final_step:.1f} 时间:{start_t}-{end_t:.1f}s", size_mb
        except Exception as e:
            return None, str(e), 0

    def _worker_video_wrapper(self, video_path: str, params: dict):
        """工作线程：包含智能压缩重试逻辑"""
        max_colors = self.config.get('gif_max_colors', 256)
        
        # 第一次尝试
        gif_io, msg, size_mb = self._process_video_core(video_path, params, max_colors)
        if not gif_io: return msg, None
        
        # 智能压缩: 如果是 GIF 且 > 10MB
        if size_mb > 10.0 and self.config.get('output_format', 'GIF').upper() == 'GIF':
            new_params = params.copy()
            new_params['scale'] = round(params['scale'] * 0.7, 2)
            if new_params['scale'] < 0.1: new_params['scale'] = 0.1
            
            retry_io, retry_msg, retry_size = self._process_video_core(video_path, new_params, 128)
            if retry_io and retry_size < size_mb:
                return f"⚠️ 原始{size_mb:.1f}MB过大，已自动压缩 -> {retry_msg}", retry_io
                
        return f"✅ 转换成功 {msg} ({size_mb:.2f}MB)", gif_io

    def _process_gif_speed(self, img_data: bytes, factor: float):
        """GIF变速处理"""
        try:
            img = PILImage.open(io.BytesIO(img_data))
            if not getattr(img, "is_animated", False): return "这不是动图", None
            
            frames, durs = [], []
            for frame in ImageSequence.Iterator(img):
                new_dur = max(20, int(frame.info.get('duration', 100) / factor))
                durs.append(new_dur)
                frames.append(frame.copy())
            
            output = io.BytesIO()
            frames[0].save(output, format='GIF', save_all=True, append_images=frames[1:], 
                         duration=durs, loop=0, disposal=2, optimize=True)
            output.seek(0)
            return "✅ 变速完成", output
        except Exception as e: return f"异常: {e}", None

    @filter.command("视频转gif")
    async def video_to_gif_cmd(self, event: AstrMessageEvent):
        """视频转GIF: /视频转gif 0s-5s fps=15"""
        params = self._parse_video_args(event.message_str.replace("视频转gif", ""))
        
        # 1. 获取源
        raw_src = self._get_media_source(event, 'video')
        if not raw_src:
            yield event.plain_result("❌ 请回复视频或发送链接")
            return
            
        # 2. 解析
        valid_src = raw_src
        if not (raw_src.startswith("http") or os.path.exists(raw_src)):
            yield event.plain_result("⏳ 解析视频地址...")
            valid_src = await self._resolve_file_via_api(event, raw_src)
            if not valid_src:
                yield event.plain_result("❌ 无法获取视频")
                return

        yield event.plain_result(f"⏳ 处理中... (缩放:{params['scale']} FPS:{params['fps']})")

        # 3. 下载与处理
        tmp_path = ""
        is_temp = False
        try:
            if valid_src.startswith("http"):
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                    tmp_path = tf.name
                    is_temp = True
                async with aiohttp.ClientSession() as sess:
                    async with sess.get(valid_src, timeout=60) as resp:
                        if resp.status != 200:
                            yield event.plain_result("❌ 下载失败")
                            return
                        with open(tmp_path, 'wb') as f: f.write(await resp.read())
            else:
                tmp_path = valid_src
            
            loop = asyncio.get_running_loop()
            msg, gif_bytes = await loop.run_in_executor(self.executor, self._worker_video_wrapper, tmp_path, params)
            
            if gif_bytes:
                yield event.chain_result([Plain(msg), Image.fromBytes(gif_bytes.getvalue())])
            else:
                yield event.plain_result(f"❌ 失败: {msg}")
        except Exception as e:
            yield event.plain_result(f"❌ 错误: {e}")
        finally:
            if is_temp and os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except: pass

    @filter.command("加速")
    async def speed_up(self, event: AstrMessageEvent, factor: str = "2"):
        await self._handle_speed(event, factor, True)

    @filter.command("减速")
    async def speed_down(self, event: AstrMessageEvent, factor: str = "2"):
        await self._handle_speed(event, factor, False)

    async def _handle_speed(self, event: AstrMessageEvent, factor_str: str, is_up: bool):
        try:
            val = float(factor_str)
            factor = val if is_up else (1.0/val)
        except: return

        img_src = self._get_media_source(event, 'image')
        # 如果是GIF图片
        if img_src and (img_src.endswith('.gif') or 'http' in img_src):
            yield event.plain_result(f"⏳ GIF变速中...")
            try:
                data = b""
                if img_src.startswith('http'):
                    async with aiohttp.ClientSession() as s:
                        async with s.get(img_src) as r: data = await r.read()
                elif os.path.exists(img_src):
                    with open(img_src, 'rb') as f: data = f.read()
                
                loop = asyncio.get_running_loop()
                msg, out = await loop.run_in_executor(self.executor, self._process_gif_speed, data, factor)
                if out: yield event.chain_result([Image.fromBytes(out.getvalue())])
                else: yield event.plain_result(f"❌ {msg}")
            except Exception as e: yield event.plain_result(f"❌ {e}")
            return
            
        yield event.plain_result("💡 若要对视频变速，请使用: /视频转gif 2x")

    # =======================================================
    # 3. 网页截图与去广告 (Playwright实现)
    # =======================================================

    async def _init_adblock_rules(self):
        if not self.config.get("screenshot_config", {}).get("enable_adblock", True): return
        if not self.rules_file.exists(): await self._update_adblock_rules()
        else: await self._load_rules_to_memory()

    async def _update_adblock_rules(self):
        urls = ["https://raw.githubusercontent.com/AdAway/adaway.github.io/master/hosts.txt"]
        content = ""
        async with aiohttp.ClientSession() as sess:
            for url in urls:
                try:
                    async with sess.get(url, timeout=10) as r:
                        if r.status==200: content += await r.text() + "\n"
                except: pass
        if content:
            with open(self.rules_file, "w", encoding="utf-8") as f: f.write(content)
            await self._load_rules_to_memory()

    async def _load_rules_to_memory(self):
        def parse():
            s = set()
            try:
                with open(self.rules_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith(('#','!')):
                            parts = line.split()
                            if len(parts)>=2: s.add(parts[1])
            except: pass
            return s
        loop = asyncio.get_running_loop()
        self.ad_domains = await loop.run_in_executor(self.executor, parse)

    async def _get_browser(self):
        async with self._browser_lock:
            if self.browser and self.browser.is_connected(): return self.browser
            if not self.playwright: self.playwright = await async_playwright().start()
            self.browser = await self.playwright.firefox.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
            return self.browser

    async def _setup_page(self, page, event: AstrMessageEvent):
        await page.set_viewport_size({"width": 1920, "height": 1080})
        # 简单拦截
        if self.config.get("screenshot_config", {}).get("enable_adblock", True):
            async def route_handler(route):
                req = route.request
                if req.resource_type in {"image", "media", "script", "xhr"}:
                    hn = urlparse(req.url).hostname
                    if hn and hn in self.ad_domains: return await route.abort()
                await route.continue_()
            await page.route("**/*", route_handler)

    @filter.command("web2img")
    async def web2img(self, event: AstrMessageEvent, url: str):
        """网页长截图: /web2img baidu.com"""
        if not url.startswith("http"): url = "https://" + url
        yield event.plain_result("⏳ 正在截取长图...")
        try:
            browser = await self._get_browser()
            page = await browser.new_page()
            try:
                await self._setup_page(page, event)
                await page.goto(url, wait_until="networkidle", timeout=60000)
                # 自动滚动
                await page.evaluate("async()=>{await new Promise(r=>{var t=0;var timer=setInterval(()=>{window.scrollBy(0,200);t+=200;if(t>=document.body.scrollHeight) {clearInterval(timer);r()}},50)})}")
                
                path = self.temp_dir / f"shot_{int(time.time())}.png"
                await page.screenshot(path=str(path), full_page=True)
                yield event.image_result(str(path))
            finally: await page.close()
        except Exception as e: yield event.plain_result(f"❌ 截图失败: {e}")

    @filter.command("web2pdf")
    async def web2pdf(self, event: AstrMessageEvent, url: str):
        """网页转PDF: /web2pdf url"""
        if not url.startswith("http"): url = "https://" + url
        yield event.plain_result("⏳ 正在转换PDF...")
        try:
            browser = await self._get_browser()
            page = await browser.new_page()
            try:
                await self._setup_page(page, event)
                await page.goto(url, wait_until="networkidle", timeout=90000)
                # 截图转PDF策略
                img_path = self.temp_dir / f"tmp_{int(time.time())}.png"
                pdf_path = self.temp_dir / f"web_{int(time.time())}.pdf"
                
                await page.screenshot(path=str(img_path), full_page=True)
                
                img = PILImage.open(str(img_path)).convert('RGB')
                img.save(str(pdf_path), "PDF", resolution=100.0)
                
                yield event.chain_result([File(file=str(pdf_path), name="WebPage.pdf")])
                os.remove(img_path)
            finally: await page.close()
        except Exception as e: yield event.plain_result(f"❌ 失败: {e}")

    # =======================================================
    # 4. OCR 功能
    # =======================================================
    @filter.command("ocr")
    async def ocr_cmd(self, event: AstrMessageEvent):
        """OCR识别: /ocr [图片]"""
        img_src = self._get_media_source(event, 'image')
        if not img_src:
            yield event.plain_result("❌ 请附带图片")
            return
            
        cfg = self.config.get("ocr_config", {})
        key = cfg.get("api_key")
        if not key:
            yield event.plain_result("未配置 OCR API Key")
            return

        try:
            img_data = b""
            if img_src.startswith("http"):
                async with aiohttp.ClientSession() as s:
                    async with s.get(img_src) as r: img_data = await r.read()
            elif os.path.exists(img_src):
                with open(img_src, "rb") as f: img_data = f.read()
            
            b64_img = base64.b64encode(img_data).decode('utf-8')
            
            api_url = cfg.get("api_url", "https://api.openai.com/v1/chat/completions")
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            payload = {
                "model": cfg.get("model_name", "gpt-4o"),
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "OCR this image. Output text only."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]}]
            }
            async with aiohttp.ClientSession() as s:
                async with s.post(api_url, headers=headers, json=payload) as r:
                    res = await r.json()
                    txt = res["choices"][0]["message"]["content"]
                    yield event.plain_result(txt)
        except Exception as e: yield event.plain_result(f"OCR Error: {e}")
