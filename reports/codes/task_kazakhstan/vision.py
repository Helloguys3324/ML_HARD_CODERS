import base64, time, os
import httpx

KEY = 'nvai-b5f01f382c5cbd6dd865cccbea63cec64014c1bbc7c98cde'
BASE = 'https://aiapi-pro.com/v1'
MODELS = ['glm-4.6v-flash', 'glm-4.1v-thinking-flash']

_last_call = [0.0]
MIN_INTERVAL = 14.0  # seconds between calls (stay under ~5/min)

def _throttle():
    wait = MIN_INTERVAL - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()

def vision(image_path=None, prompt='describe', image_bytes=None, retries=6, model=None):
    if image_bytes is not None:
        b64 = base64.b64encode(image_bytes).decode()
        mime = 'image/jpeg'
    else:
        ext = os.path.splitext(image_path)[1].lower()
        mime = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png', 'webp': 'image/webp'}.get(ext, 'image/jpeg')
        b64 = base64.b64encode(open(image_path, 'rb').read()).decode()
    url = f"data:{mime};base64,{b64}"
    ms = MODELS if model is None else [model]
    last_err = None
    for attempt in range(retries):
        _throttle()
        for m in ms:
            payload = {'model': m, 'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': url}},
            ]}], 'max_tokens': 900}
            try:
                r = httpx.post(BASE + '/chat/completions', headers={'Authorization': 'Bearer ' + KEY}, json=payload, timeout=240)
            except Exception as e:
                last_err = str(e)
                continue
            if r.status_code == 200:
                return r.json()['choices'][0]['message']['content']
            last_err = r.text[:200]
            if r.status_code == 402:
                return {'error': last_err}
        time.sleep(20)
    return {'error': last_err}

if __name__ == '__main__':
    import sys
    print(vision(sys.argv[1], sys.argv[2]))