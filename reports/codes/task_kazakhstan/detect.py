import re
import sys
import cv2
sys.path.insert(0, '/home/kozak/mllm')
from challenges.task_kazakhstan.vision import vision

PROMPT = """This is a scan of an album page (1600x1200 px) showing three Kazakhstan postage stamps on a dark background. Exactly ONE of the three stamps has a small forged/altered region (a digit changed, an element cloned or duplicated, or an inscription removed). Inspect carefully and find it.
Respond with ONLY one JSON line and nothing else:
{"stamp": 1, "type": "digit_changed", "cx": 100, "cy": 200, "w": 40, "h": 30}
where "stamp" is the index reading the stamps in order (1,2,3), and cx,cy,w,h are the bounding box of the altered region in FULL PAGE pixel coordinates (origin top-left of the page)."""


def detect_page(path):
    raw = vision(path, PROMPT, model='glm-4.1v-thinking-flash')
    text = raw if isinstance(raw, str) else str(raw)
    # find JSON substring
    m = re.search(r'\{[^{}]*\}', text)
    if m:
        try:
            import json
            return json.loads(m.group(0))
        except Exception:
            pass
    return {'raw': text}


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'challenges/task_kazakhstan/data/albums/album_1/page_1759.jpg'
    print(detect_page(path))