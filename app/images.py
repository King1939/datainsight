"""文档内嵌图片与独立图片处理：规范化、编码与步骤级收集。

设计要点：
  1. Pillow (PIL) 为可选增强：有它时能处理 CMYK/BMP/TIFF 转码、EXIF 纠正与大图缩放；
     没装时自动退回魔数嗅探，原生 PNG/JPEG/GIF/WEBP 照样能正常提交。
  2. 图片只活在当轮送往模型的 `request_messages` 里，不把大块 Base64 塞进
     `session.messages` 或 `data/chats/*.json`——既省盘，也避免切换纯文本模型后历史报 400。
  3. 工具返回值契约依然是纯 `str`：通过 `ContextVar` 旁路收集本步读取到的图片，
     与 `data_tools._ACTIVE` 一样的线程/上下文隔离机制。
"""

from __future__ import annotations

import base64
import contextlib
import contextvars
import hashlib
import io
import struct
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Sequence

from .config import IMAGE_MAX_BYTES, IMAGE_MAX_SIDE, IMAGE_MAX_PER_TURN

SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")

_EXT_BY_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


def ext_for_mime(mime_type: str) -> str:
    return _EXT_BY_MIME.get((mime_type or "").lower(), ".png")


@dataclass(frozen=True)
class DocImage:
    """从文档或独立图片文件中抽出来的一张规范化图片。"""

    id: str
    name: str
    mime_type: str
    width: int = 0
    height: int = 0
    part: int = 1
    location: str = ""
    size: int = 0
    path: str = ""
    data: bytes = field(default=b"", repr=False, compare=False)

    def with_part(self, part: int) -> "DocImage":
        return replace(self, part=max(1, int(part)))

    def read_bytes(self) -> bytes:
        if self.data:
            return self.data
        if self.path:
            try:
                return Path(self.path).read_bytes()
            except OSError:
                return b""
        return b""

    def to_data_url(self) -> str:
        raw = self.read_bytes()
        if not raw:
            return ""
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"

    def to_content_part(self) -> dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": self.to_data_url()},
        }

    def to_meta(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "mime_type": self.mime_type,
            "width": self.width,
            "height": self.height,
            "part": self.part,
            "location": self.location,
            "size": self.size,
        }

    @classmethod
    def from_meta(cls, meta: dict[str, Any], *, path: str = "", data: bytes = b"") -> "DocImage":
        return cls(
            id=str(meta.get("id") or ""),
            name=str(meta.get("name") or "image.png"),
            mime_type=str(meta.get("mime_type") or "image/png"),
            width=int(meta.get("width") or 0),
            height=int(meta.get("height") or 0),
            part=max(1, int(meta.get("part") or 1)),
            location=str(meta.get("location") or ""),
            size=int(meta.get("size") or len(data) or 0),
            path=path,
            data=data,
        )


def _sniff_image(raw: bytes) -> tuple[str, int, int] | None:
    """无 Pillow 时的轻量魔数嗅探（PNG / JPEG / GIF / WEBP）。"""
    if len(raw) < 12:
        return None
    if raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) >= 24:
        w, h = struct.unpack(">II", raw[16:24])
        return "image/png", w, h
    if raw[:6] in (b"GIF87a", b"GIF89a") and len(raw) >= 10:
        w, h = struct.unpack("<HH", raw[6:10])
        return "image/gif", w, h
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp", 0, 0
    if raw[:2] == b"\xff\xd8":
        w, h = _sniff_jpeg_dims(raw)
        return "image/jpeg", w, h
    return None


def _sniff_jpeg_dims(raw: bytes) -> tuple[int, int]:
    idx = 2
    n = len(raw)
    while idx + 4 < n:
        if raw[idx] != 0xFF:
            idx += 1
            continue
        marker = raw[idx + 1]
        if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
            idx += 2
            continue
        if idx + 4 > n:
            break
        seg_len = struct.unpack(">H", raw[idx + 2 : idx + 4])[0]
        if seg_len < 2:
            break
        if marker in (0xC0, 0xC1, 0xC2, 0xC3) and idx + 9 <= n:
            h, w = struct.unpack(">HH", raw[idx + 5 : idx + 9])
            return w, h
        idx += 2 + seg_len
    return 0, 0


def _ensure_name(name: str, img_id: str, mime_type: str) -> str:
    ext = ext_for_mime(mime_type)
    clean = Path(str(name or "").replace("\\", "/")).name.strip()
    if not clean:
        return f"img_{img_id[:8]}{ext}"
    stem = Path(clean).stem or f"img_{img_id[:8]}"
    return f"{stem}{ext}"


def _encode_with_pil(
    raw: bytes,
    *,
    max_side: int,
    max_bytes: int,
) -> tuple[bytes, str, int, int] | None:
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return None

    try:
        with Image.open(io.BytesIO(raw)) as opened:
            img = ImageOps.exif_transpose(opened)
            w, h = img.size
            if max(w, h) <= 2:
                return None

            if max(w, h) > max_side:
                ratio = max_side / float(max(w, h))
                new_size = (max(1, int(round(w * ratio))), max(1, int(round(h * ratio))))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                w, h = img.size

            src_fmt = (opened.format or "").upper()
            has_alpha = "A" in img.getbands() or (
                img.mode == "P" and "transparency" in img.info
            )
            if has_alpha:
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # GIF 仅静帧或直接转 PNG；原始是 JPEG 且转码后非透明时保留 JPEG 更省体积
            if src_fmt == "JPEG" and not has_alpha:
                return _compress_jpeg(img, max_bytes=max_bytes)

            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            png_bytes = buf.getvalue()
            if len(png_bytes) <= max_bytes:
                return png_bytes, "image/png", w, h

            # 超出体积上限时转 JPEG（若含透明先铺白底）再逐级压缩
            rgb = _to_rgb_white_bg(img) if img.mode == "RGBA" else img
            return _compress_jpeg(rgb, max_bytes=max_bytes)
    except Exception:  # noqa: BLE001  坏图 / 不支持解码的 EMF-WMF 静默跳过
        return None


def _to_rgb_white_bg(img):
    from PIL import Image

    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[-1])
    return bg


def _compress_jpeg(img, *, max_bytes: int) -> tuple[bytes, str, int, int] | None:
    from PIL import Image

    work = img if img.mode == "RGB" else img.convert("RGB")
    for scale in (1.0, 0.75, 0.5):
        candidate = work
        if scale < 1.0:
            nw = max(1, int(work.width * scale))
            nh = max(1, int(work.height * scale))
            candidate = work.resize((nw, nh), Image.Resampling.LANCZOS)
        for quality in (85, 70, 55):
            buf = io.BytesIO()
            candidate.save(buf, format="JPEG", quality=quality, optimize=True)
            data = buf.getvalue()
            if len(data) <= max_bytes:
                return data, "image/jpeg", candidate.width, candidate.height
    return None


def normalize_image(
    raw: bytes,
    *,
    name: str = "",
    part: int = 1,
    location: str = "",
    max_side: int = IMAGE_MAX_SIDE,
    max_bytes: int = IMAGE_MAX_BYTES,
) -> DocImage | None:
    """把原始图片字节规范化为 OpenAI Vision 可接受的 PNG/JPEG/WEBP/GIF。"""
    if not raw:
        return None

    encoded = _encode_with_pil(raw, max_side=max_side, max_bytes=max_bytes)
    if encoded is not None:
        out_bytes, mime_type, w, h = encoded
    else:
        sniffed = _sniff_image(raw)
        if sniffed is None or len(raw) > max_bytes:
            return None
        mime_type, w, h = sniffed
        if w and h and max(w, h) <= 2:
            return None
        out_bytes = raw

    img_id = hashlib.sha1(out_bytes).hexdigest()[:16]
    return DocImage(
        id=img_id,
        name=_ensure_name(name, img_id, mime_type),
        mime_type=mime_type,
        width=w,
        height=h,
        part=max(1, int(part)),
        location=location,
        size=len(out_bytes),
        data=out_bytes,
    )


# ---------------------------------------------------------------- 步骤级图片收集器


_COLLECTOR: contextvars.ContextVar[list[DocImage] | None] = contextvars.ContextVar(
    "datainsight.step_images", default=None
)


@contextlib.contextmanager
def collect_step_images() -> Iterator[list[DocImage]]:
    """在单步工具调用窗口内收集附着的图片（线程/会话安全隔离）。"""
    bucket: list[DocImage] = []
    token = _COLLECTOR.set(bucket)
    try:
        yield bucket
    finally:
        _COLLECTOR.reset(token)


def attach_step_images(
    items: Sequence[DocImage], *, limit: int = IMAGE_MAX_PER_TURN
) -> list[DocImage]:
    """工具把本段图片登记进当前步骤；返回实际附着（去重限流后）的图片列表。"""
    cap = max(0, int(limit))
    bucket = _COLLECTOR.get()
    seen_ids = {img.id for img in bucket} if bucket is not None else set()
    attached: list[DocImage] = []
    for img in items:
        if not img.read_bytes():
            continue
        if img.id in seen_ids:
            continue
        if cap and len(attached) >= cap:
            break
        seen_ids.add(img.id)
        attached.append(img)
        if bucket is not None:
            bucket.append(img)
    return attached


__all__ = [
    "DocImage",
    "SUPPORTED_IMAGE_SUFFIXES",
    "attach_step_images",
    "collect_step_images",
    "ext_for_mime",
    "normalize_image",
]
