import torch, base64, os, sys
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

MODEL_ID = 'Qwen/Qwen2.5-VL-3B-Instruct'
_model = None
_processor = None

def get_model():
    global _model, _processor
    if _model is None:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32, device_map='cpu')
        _model.eval()
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
    return _model, _processor

def ask(image_path, prompt, max_new_tokens=512, max_pixels=1024*1024, image=None):
    model, proc = get_model()
    if image is None:
        image = Image.open(image_path).convert('RGB')
    messages = [{'role': 'user', 'content': [
        {'type': 'image'},
        {'type': 'text', 'text': prompt},
    ]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[image], padding=True, return_tensors='pt')
    # limit pixels
    if 'pixel_values' in inputs:
        pass
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, inputs['input_ids'].shape[1]:]
    ans = proc.batch_decode(gen, skip_special_tokens=True)[0]
    return ans

if __name__ == '__main__':
    import time
    t = time.time()
    print(ask(sys.argv[1], sys.argv[2], max_new_tokens=int(sys.argv[3]) if len(sys.argv) > 3 else 512))
    print(f'[time {time.time()-t:.1f}s]', file=sys.stderr)