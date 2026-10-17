"""Smoke test of the Step5 DSA kernels on SM120 (one GPU, no model weights).

Each test JIT-compiles one kernel the Step5 vLLM path uses, runs it on small
random inputs and compares against a plain PyTorch reference. Run it inside
the vllm/vllm-openai:step-5-preview image with the patched files mounted.

    python3 smoke_sm120.py
"""

import math
import sys
import traceback

import torch

DEV = torch.device("cuda:0")
torch.manual_seed(0)
RESULTS = []


def check(name, out, ref, atol, rtol):
    out, ref = out.float(), ref.float()
    err = (out - ref).abs()
    tol = atol + rtol * ref.abs()
    bad = (err > tol).sum().item()
    ok = bad == 0 and torch.isfinite(out).all().item()
    print(f"    max|err|={err.max().item():.3e}  out_of_tol={bad}")
    if not ok:
        raise AssertionError(f"{name}: {bad} values out of tolerance")


def run(name, fn):
    print(f"\n=== {name}")
    try:
        fn()
        RESULTS.append((name, "OK"))
        print("    -> OK")
    except Exception as e:
        RESULTS.append((name, f"FAIL: {type(e).__name__}: {str(e)[:200]}"))
        traceback.print_exc()
        print("    -> FAIL")


def fp8(t):
    return t.clamp(-448, 448).to(torch.float8_e4m3fn)


# ---------------------------------------------------------------- 0. environment
def t_env():
    import triton

    import vllm

    cap = torch.cuda.get_device_capability(DEV)
    props = torch.cuda.get_device_properties(DEV)
    smem = getattr(props, "shared_memory_per_block_optin", None)
    print(f"    GPU={props.name} cap={cap} SMs={props.multi_processor_count}")
    print(f"    smem/block (optin)={smem}  torch={torch.__version__}")
    print(f"    triton={triton.__version__}  vllm={vllm.__version__}")
    from vllm.models.step5.nvidia.sparse_attention import Step5DSAAttentionBackend
    from vllm.platforms.interface import DeviceCapability

    if not Step5DSAAttentionBackend.supports_compute_capability(
        DeviceCapability(*cap)
    ):
        raise RuntimeError("The capability patch is not mounted")


# ------------------------------------------------------------- 1. RMSNorm CuTe
def t_rmsnorm():
    from vllm.models.step5.nvidia.layernorm import _step5_rms_norm
    from vllm.models.step5.nvidia.ops.rms_norm import rms_norm

    x = torch.randn(7, 4096, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(4096, device=DEV, dtype=torch.bfloat16) * 0.1
    out = rms_norm(x, w, 1e-6, weight_bias=1.0)
    ref = _step5_rms_norm(x, w, 1e-6, True)
    check("rmsnorm", out, ref, atol=2e-2, rtol=2e-2)
    r = torch.randn(7, 4096, device=DEV, dtype=torch.float32)
    r_ref = x.float() + r
    out2 = rms_norm(x, w, 1e-6, weight_bias=1.0, residual=r)
    check("rmsnorm+residual", out2, _step5_rms_norm(r_ref, w, 1e-6, True, torch.bfloat16),
          atol=2e-2, rtol=2e-2)
    check("residual in-place", r, r_ref, atol=1e-5, rtol=1e-5)


# ------------------------------------------ 2. indexer logits (prefill)
def t_paged_logits():
    from vllm.models.step5.nvidia.ops.sparse_indexer_scores import (
        paged_mqa_logits_sm90,
    )

    B, N, H, D, P = 2, 2, 8, 256, 32
    regions = 300
    pages_per_req = math.ceil(regions / P)
    num_pages = B * pages_per_req + 3
    q = fp8(torch.randn(B, N, H, D, device=DEV) * 0.5)
    keys = fp8(torch.randn(num_pages, P, D, device=DEV) * 0.5)
    scales = torch.rand(num_pages, P, device=DEV) + 0.5
    kv = torch.empty(num_pages, P * 260, dtype=torch.uint8, device=DEV)
    kv[:, : P * 256] = keys.view(torch.uint8).reshape(num_pages, -1)
    kv[:, P * 256 :] = scales.contiguous().view(torch.uint8).reshape(num_pages, -1)
    kv = kv.view(num_pages, P, 1, 260)
    w = torch.rand(B * N, H, device=DEV)
    lengths = torch.tensor([regions, 137], dtype=torch.int32, device=DEV)
    table = torch.randperm(num_pages, device=DEV)[: B * pages_per_req]
    table = table.view(B, pages_per_req).int()

    out = paged_mqa_logits_sm90(q, kv, w, lengths, table, regions)

    ref = torch.full((B * N, regions), float("-inf"), device=DEV)
    for b in range(B):
        L = int(lengths[b])
        r = torch.arange(L, device=DEV)
        phys = table[b, r // P].long()
        k = keys[phys, r % P].float()  # [L, D]
        s = scales[phys, r % P]
        for n in range(N):
            dots = torch.relu(k @ q[b, n].float().T)  # [L, H]
            ref[b * N + n, :L] = (dots * w[b * N + n]).sum(-1) * s
    finite = torch.isfinite(ref)
    if not torch.equal(torch.isfinite(out), finite):
        raise AssertionError("-inf mask differs")
    check("paged_logits", out[finite], ref[finite], atol=1e-3, rtol=1e-4)


# ------------------------------------------ 3. indexer logits (decode)
def t_decode_scores():
    from vllm.models.step5.nvidia.ops.sparse_indexer_scores import (
        decode_summary_scores,
    )

    T, H, D, RPB = 4, 8, 256, 8  # block_size 64 -> 8 regions per block
    cols, pages = 64, 300
    capacity = cols * RPB
    seq = torch.tensor([1, 500, 4000, 3333], dtype=torch.int32, device=DEV)
    q = fp8(torch.randn(T, 1, H, D, device=DEV) * 0.5)
    w = torch.rand(T, 1, H, device=DEV)
    mean = fp8(torch.randn(pages, RPB, D, device=DEV) * 0.5).view(torch.uint8)
    table = torch.stack([torch.randperm(pages, device=DEV)[:cols] for _ in range(T)])
    table = table.int().contiguous()
    slots = torch.arange(T, device=DEV, dtype=torch.int64)
    props = torch.cuda.get_device_properties(DEV)

    scores, meta = decode_summary_scores(
        q, w, mean, table, slots, seq, capacity, num_sms=props.multi_processor_count
    )
    torch.cuda.synchronize()
    for t in range(T):
        length = (int(seq[t]) - 1) // 8 + 1
        r = torch.arange(length, device=DEV)
        phys = table[t, r // RPB].long()
        k = mean[phys, r % RPB].view(torch.float8_e4m3fn).float()
        ref = (torch.relu(k @ q[t, 0].float().T) * w[t, 0]).sum(-1)
        check(f"decode_scores row {t}", scores[t, :length], ref, atol=1e-3, rtol=1e-4)
        if int(meta.lengths[t]) != length:
            raise AssertionError("wrong metadata lengths")


# ------------------------------------------------ 4. region top-k
def t_select():
    from vllm.models.step5.nvidia.ops.sparse_indexer import select_regions

    rows, regions, topk = 3, 200, 64
    scores = torch.randn(rows, regions, device=DEV)
    current = torch.tensor([150, 199, 40], device=DEV, dtype=torch.int64)
    out = select_regions(scores, current, topk, "auto", sort=True)
    torch.cuda.synchronize()
    for i in range(rows):
        c = int(current[i])
        row = out[i].tolist()
        if row[-1] != c:
            raise AssertionError(f"row {i}: last entry is not the current region")
        got = sorted(x for x in row[:-1] if x >= 0)
        if c <= topk:
            want = list(range(c))
        else:
            want = sorted(torch.topk(scores[i, :c], topk).indices.tolist())
        if got != want:
            raise AssertionError(f"row {i}: selection differs from reference")
    print("    selection identical to torch.topk")


# ------------------------------------------- 5. sparse decode attention
def _decode_case(splits_override):
    from vllm.models.step5.nvidia.ops.sparse_decode_attention import (
        sparse_decode_attention,
        sparse_decode_splits,
    )

    rows, H, BS, PPR, R = 2, 8, 64, 4, 16
    pages = rows * PPR + 2
    cache = torch.randn(pages, 1, BS, 384, device=DEV, dtype=torch.bfloat16)
    table = torch.randperm(pages, device=DEV)[: rows * PPR].view(rows, PPR).int()
    q = torch.randn(rows, H, 192, device=DEV, dtype=torch.bfloat16)
    positions = torch.tensor([200, 77], device=DEV, dtype=torch.int64)
    active = torch.ones(rows, dtype=torch.bool, device=DEV)
    regions = torch.full((rows, R), -1, dtype=torch.long, device=DEV)
    cur0 = 200 // 8
    hist0 = torch.randperm(cur0, device=DEV)[: R - 1].sort().values
    regions[0, :-1], regions[0, -1] = hist0, cur0
    regions[1, R - 10 :] = torch.arange(10, device=DEV)  # 77//8+1 = 10 live regions
    out = torch.empty_like(q)
    scale = 1 / math.sqrt(192)
    splits = splits_override or sparse_decode_splits(rows)
    sparse_decode_attention(
        q, cache, regions, table, positions, active, out,
        scale=scale, splits=splits,
    )
    torch.cuda.synchronize()
    for r in range(rows):
        sel = regions[r][regions[r] >= 0]
        tok = (sel[:, None] * 8 + torch.arange(8, device=DEV)).flatten()
        tok = tok[tok <= positions[r]]
        kv = cache[table[r, tok // BS].long(), 0, tok % BS].float()
        k, v = kv[:, :192], kv[:, 192:]
        p = torch.softmax((q[r].float() @ k.T) * scale, -1)
        check(f"decode attn row {r} splits={splits}", out[r], p @ v,
              atol=2e-2, rtol=2e-2)


def t_decode_attn():
    _decode_case(1)
    _decode_case(None)


# ------------------------- 6. FlashInfer FA2 paged, head_dim 192 (prefill)
def t_flashinfer_paged():
    import flashinfer

    H, D, PS, npages, nq = 8, 192, 8, 5, 6
    ws = torch.empty(128 << 20, dtype=torch.uint8, device=DEV)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
    k = torch.randn(npages, PS, 1, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(npages, PS, 1, D, device=DEV, dtype=torch.bfloat16)
    q = torch.randn(nq, H, D, device=DEV, dtype=torch.bfloat16)
    idx = torch.tensor([3, 0, 4], dtype=torch.int32, device=DEV)
    scale = 1 / math.sqrt(D)
    wrapper.plan(
        torch.tensor([0, nq], dtype=torch.int32),
        torch.tensor([0, 3], dtype=torch.int32),
        idx,
        torch.tensor([5], dtype=torch.int32),
        H, 1, D, PS,
        head_dim_vo=D, causal=False, sm_scale=scale,
        q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
    )
    out = wrapper.run(q, (k, v))
    kk = torch.cat([k[3, :, 0], k[0, :, 0], k[4, :5, 0]]).float()
    vv = torch.cat([v[3, :, 0], v[0, :, 0], v[4, :5, 0]]).float()
    ref = torch.softmax(q.float() @ kk.T * scale, -1) @ vv
    check("flashinfer paged fa2 d192", out, ref, atol=2e-2, rtol=2e-2)


# ------------------------- 7. FlashInfer FA2 ragged, head_dim 192 (prefill)
def t_flashinfer_ragged():
    import flashinfer

    H, D, nq, nk = 8, 192, 5, 37
    ws = torch.empty(128 << 20, dtype=torch.uint8, device=DEV)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="fa2")
    q = torch.randn(nq, H, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(nk, 1, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(nk, 1, D, device=DEV, dtype=torch.bfloat16)
    scale = 1 / math.sqrt(D)
    wrapper.plan(
        torch.tensor([0, nq], dtype=torch.int32),
        torch.tensor([0, nk], dtype=torch.int32),
        H, 1, D, head_dim_vo=D, causal=False, sm_scale=scale,
        q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
        disable_split_kv=True,
    )
    out = wrapper.run(q, k, v)
    ref = torch.softmax(q.float() @ k[:, 0].float().T * scale, -1) @ v[:, 0].float()
    check("flashinfer ragged fa2 d192", out, ref, atol=2e-2, rtol=2e-2)


# ---------------- 8. vLLM FlashAttention 2 paged d192 (sliding-window layers)
def t_vllm_fa2():
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    H, KVH, D, BS, nblocks = 8, 1, 192, 64, 6
    k = torch.randn(nblocks, BS, KVH, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(nblocks, BS, KVH, D, device=DEV, dtype=torch.bfloat16)
    seqlen, nq = 150, 4
    table = torch.tensor([[2, 5, 0]], dtype=torch.int32, device=DEV)
    q = torch.randn(nq, H, D, device=DEV, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    scale = 1 / math.sqrt(D)
    flash_attn_varlen_func(
        q=q, k=k, v=v, out=out,
        cu_seqlens_q=torch.tensor([0, nq], dtype=torch.int32, device=DEV),
        seqused_k=torch.tensor([seqlen], dtype=torch.int32, device=DEV),
        max_seqlen_q=nq, max_seqlen_k=seqlen, softmax_scale=scale,
        causal=True, window_size=[-1, -1], block_table=table, fa_version=2,
    )
    kk = torch.cat([k[i, :, 0] for i in (2, 5, 0)])[:seqlen].float()
    vv = torch.cat([v[i, :, 0] for i in (2, 5, 0)])[:seqlen].float()
    s = q.float() @ kk.T * scale  # [nq, H, seqlen]
    qpos = torch.arange(seqlen - nq, seqlen, device=DEV)
    mask = torch.arange(seqlen, device=DEV)[None, :] > qpos[:, None]
    s = s.masked_fill(mask[:, None, :], float("-inf"))
    ref = torch.softmax(s, -1) @ vv
    check("vllm fa2 paged d192", out, ref, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    run("0. environment and capability patch", t_env)
    run("1. RMSNorm (CuTe DSL) [non-blocking: PyTorch fallback exists]", t_rmsnorm)
    run("2. indexer logits, prefill (Triton FP8)", t_paged_logits)
    run("3. indexer logits, decode (Triton FP8)", t_decode_scores)
    run("4. region top-k (per_row)", t_select)
    run("5. sparse decode attention (Triton)", t_decode_attn)
    run("6. FlashInfer FA2 paged d192", t_flashinfer_paged)
    run("7. FlashInfer FA2 ragged d192", t_flashinfer_ragged)
    run("8. vLLM FlashAttention2 paged d192", t_vllm_fa2)
    print("\n================ SUMMARY ================")
    for name, status in RESULTS:
        print(f"{status[:2]:>3}  {name}" + ("" if status == "OK" else f"\n      {status}"))
    sys.exit(0 if all(s == "OK" for _, s in RESULTS) else 1)
