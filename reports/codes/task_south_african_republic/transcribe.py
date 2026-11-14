"""Transcribe clean.wav with faster-whisper, chunked multiprocessing, multi-language.

Usage: transcribe.py <model> <lang1,lang2,...|None> <workers> <threads_per_worker>
Languages: comma list or 'None' for auto-detect.
Appends JSONL segments to transcript.jsonl (with 'forced' tag) and prints text.
"""
import json
import sys
import numpy as np
import soundfile as sf

WAV = "/home/kozak/mllm/challenges/task_south_african_republic/clean.wav"
OUT = "/home/kozak/mllm/challenges/task_south_african_republic/transcript.jsonl"

MODEL = sys.argv[1] if len(sys.argv) > 1 else "small"
LANGS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["None"]
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
THREADS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
CHUNK = 45
OVERLAP = 6

_MODEL = None


def _init_worker(model_name, cthreads):
    global _MODEL
    from faster_whisper import WhisperModel
    _MODEL = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=cthreads)


def _globals(model_name, cthreads):
    _init_worker(model_name, cthreads)


def _do(args):
    start, audio, lang = args
    language = None if lang == "None" else lang
    segs, info = _MODEL.transcribe(
        audio, language=language, vad_filter=False, beam_size=5,
        no_speech_threshold=None, log_prob_threshold=None,
        condition_on_previous_text=False,
    )
    res = []
    for s in segs:
        res.append({
            "start": round(start + s.start, 2),
            "end": round(start + s.end, 2),
            "text": s.text.strip(),
            "lang_det": info.language,
            "lang_prob": round(info.language_probability, 3),
            "avg_logprob": round(s.avg_logprob, 3),
            "model": MODEL,
            "forced": lang,
        })
    return res


def main():
    audio, sr = sf.read(WAV, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n = len(audio)

    jobs = []
    step = CHUNK - OVERLAP
    t = 0.0
    while True:
        s0 = int(t * sr)
        s1 = min(n, int((t + CHUNK) * sr))
        for lang in LANGS:
            jobs.append((round(t, 2), audio[s0:s1], lang))
        if s1 >= n:
            break
        t += step

    print(f"[{MODEL}] langs={LANGS} chunks*{len(LANGS)}={len(jobs)} workers={WORKERS} threads={THREADS}", flush=True)

    all_out = []
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(WORKERS, initializer=_globals, initargs=(MODEL, THREADS)) as pool:
        for i, res in enumerate(pool.imap_unordered(_do, jobs)):
            all_out.extend(res)
            if (i + 1) % max(1, len(jobs) // 10) == 0:
                print(f"  {i+1}/{len(jobs)} jobs done", flush=True)

    all_out.sort(key=lambda x: (x["start"], x["forced"]))
    with open(OUT, "a") as f:
        for r in all_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 70)
    for r in all_out:
        print(f"[{r['start']:7.1f}-{r['end']:7.1f}|{r['forced']:>5}] {r['text']}", flush=True)


if __name__ == "__main__":
    main()