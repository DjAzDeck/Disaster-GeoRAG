import torch
from PIL import Image
import os, warnings
warnings.filterwarnings("ignore")

# RS-LLaVA imports (repo cloned locally)
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

class RSLLaVA:
    def __init__(self,
                 model_path="BigData-KSU/RS-llava-v1.5-7b-LoRA",
                 model_base="Intel/neural-chat-7b-v3-3",
                 conv_mode="llava_v1",
                 temperature=0.2,
                 max_new_tokens=512,
                 device=None):
        self.model_path=model_path; self.model_base=model_base
        self.conv_mode=conv_mode; self.temperature=temperature; self.max_new_tokens=max_new_tokens
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, model_name
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def _build_input(self, prompt: str):
        if self.model.config.mm_use_im_start_end:
            return f"{DEFAULT_IM_START_TOKEN} {DEFAULT_IMAGE_TOKEN} {DEFAULT_IM_END_TOKEN}\n{prompt}"
        else:
            return f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"

    @torch.inference_mode()
    def infer_json(self, image, prompt: str):
        if isinstance(image, (str, bytes)):
            im = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            im = image.convert("RGB")
        else:
            raise TypeError("image must be a file path or PIL.Image")

        img_tensor = self.image_processor.preprocess(im, return_tensors='pt')['pixel_values'][0]
        conv = conv_templates[self.conv_mode].copy()
        cur_prompt = self._build_input(prompt)
        conv.append_message(conv.roles[0], cur_prompt)
        conv.append_message(conv.roles[1], None)
        final = conv.get_prompt()

        input_ids = tokenizer_image_token(final, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        outputs = self.model.generate(
            input_ids,
            images=img_tensor.unsqueeze(0).to(self.device, dtype=torch.float16 if self.device=='cuda' else torch.float32),
            do_sample=True, temperature=self.temperature, num_beams=1,
            max_new_tokens=self.max_new_tokens, use_cache=True
        )
        txt = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        return txt
