"""推理基准：吞吐 / 首 token 延迟 / 峰值显存。与质量评测解耦，可独立跑。"""
import time

import torch


@torch.no_grad()
def bench_throughput(model, tok, prompts, max_new_tokens: int = 64, warmup: int = 2) -> dict:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    decode_tps, prefill_tps, ttft = [], [], []

    def run(p):
        inputs = tok(p, return_tensors="pt").to(model.device)
        n_in = inputs["input_ids"].shape[1]
        t0 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        t1 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        t2 = time.perf_counter()
        n_out = out.shape[1] - n_in
        ttft.append(t1 - t0)
        if n_out > 1 and (t2 - t1) > 0:
            decode_tps.append((n_out - 1) / (t2 - t1))
        if (t1 - t0) > 0:
            prefill_tps.append(n_in / (t1 - t0))
        return n_out

    for p in prompts[:warmup]:
        run(p)
    for p in prompts:
        run(p)

    res = {
        "n_prompts": len(prompts),
        "max_new_tokens": max_new_tokens,
        "decode_tokens_per_sec": round(sum(decode_tps) / len(decode_tps), 2) if decode_tps else 0.0,
        "prefill_tokens_per_sec": round(sum(prefill_tps) / len(prefill_tps), 2) if prefill_tps else 0.0,
        "ttft_ms": round(1000 * sum(ttft) / len(ttft), 2) if ttft else 0.0,
    }
    if torch.cuda.is_available():
        res["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    return res
