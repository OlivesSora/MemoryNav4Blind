"""Extract visible text from anchor images using a remote vision-language model.

Uses the ModelScope OpenAI-compatible endpoint and the ``Qwen/Qwen3-VL-8B-Instruct``
vision model.  The API key pool is read from a JSON file (defaults to the L4
project's ``config copy/api_keys.json``).  If the key file is missing or the API
is unreachable, ``extract`` degrades gracefully and returns ``""``.
"""

from __future__ import annotations

import base64
import io
import json
import random
from pathlib import Path

import numpy as np

MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_API_KEY_FILE = Path(
    "/home/wheeltec/projects/L4/reFineWithMacroAndMicroLocal/config copy/api_keys.json"
)

PROMPT = (
    "请识别并输出这张图像中出现的所有文字/文本（如路牌、店名、招牌、门牌号、箭头提示语等），"
    "保留原有文字内容。只输出识别到的文本，用顿号或空格分隔每一项；"
    "若图像中没有文字，只输出：无"
)


class VisionTextExtractor:
    def __init__(self, api_key_file: str | Path | None = None,
                 model: str = DEFAULT_MODEL, max_side: int = 640):
        self.model = model
        self.max_side = max_side
        self._client = None
        self._error: str | None = None
        self._load_client(api_key_file if api_key_file is not None else DEFAULT_API_KEY_FILE)

    def _load_client(self, api_key_file: str | Path) -> None:
        try:
            path = Path(api_key_file).expanduser()
            if not path.is_file():
                self._error = f"API key file not found: {path}"
                return
            keys = json.loads(path.read_text(encoding="utf-8"))
            keys = [k for k in keys if isinstance(k, str) and k.strip()]
            if not keys:
                self._error = "API key file is empty"
                return
            from openai import OpenAI

            self._client = OpenAI(
                api_key=random.choice(keys),
                base_url=MODELSCOPE_BASE_URL,
                timeout=300,
            )
        except Exception as exc:
            self._error = str(exc)

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def error(self) -> str | None:
        return self._error

    def extract(self, frame_bgr: np.ndarray) -> str:
        if self._client is None:
            return ""
        try:
            import cv2
            from PIL import Image

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            width, height = image.size
            ratio = min(1.0, self.max_side / max(width, height))
            if ratio < 1.0:
                image = image.resize((int(width * ratio), int(height * ratio)))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            data_uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }]
            response = self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.0, max_tokens=512
            )
            text = (response.choices[0].message.content or "").strip()
            if not text or text in ("无", "没有", "暂无", "无文字", "没有文字"):
                return ""
            return text
        except Exception:
            return ""