from dataclasses import dataclass
from typing import Union
import torch
from PIL import Image
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

@dataclass
class Qwen2VLConfig:
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    max_new_tokens: int = 192
    dtype: str = "auto"  # "auto" | "bfloat16" | "float16"

class Qwen2VL:
    def __init__(self, cfg: Qwen2VLConfig | None = None):
        self.cfg = cfg or Qwen2VLConfig()
        dtype = {"auto":"auto","bfloat16":torch.bfloat16,"float16":torch.float16}[self.cfg.dtype]
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.cfg.model_id, device_map="auto", torch_dtype=dtype
        )
        self.processor = AutoProcessor.from_pretrained(
            self.cfg.model_id,
            use_fast=False,
            min_pixels=224*224,
            max_pixels=2048*2048
        )

    @torch.inference_mode()
    def infer_json(self, image: Union[str, Image.Image], prompt_text: str) -> str:
        """Accepts str path, PIL.Image, NumPy array, or Gradio-style dict; returns raw text."""
        def _to_pil(x):

            if isinstance(x, (str, bytes)):
                return Image.open(x).convert("RGB")

            if isinstance(x, Image.Image):
                return x.convert("RGB")

            if isinstance(x, dict):
                for k in ("image", "array", "value"):
                    if k in x and x[k] is not None:
                        return _to_pil(x[k])
                raise TypeError("Unsupported dict payload for image")

            try:
                import numpy as np
                arr = np.array(x)
                if arr.ndim == 2:
                    arr = np.stack([arr]*3, axis=-1)
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255)
                    arr = arr.astype(np.uint8)
                return Image.fromarray(arr, mode="RGB")
            except Exception as e:
                raise TypeError("image must be a file path, PIL.Image, array-like, or dict") from e

        image = _to_pil(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(messages, image_patch_size=14)

        model_inputs = self.processor(
            text=text, images=images, videos=videos,
            do_resize=False, return_tensors="pt"
        ).to(self.model.device)

        out = self.model.generate(
            **model_inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=False, temperature=0.0, top_p=1.0, use_cache=True
        )
        result = self.processor.batch_decode(
            out[:, model_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()
        return result
