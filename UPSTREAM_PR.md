# [Step5][SM120] Enable Step-5 DSA on SM12x (RTX PRO 6000 / RTX 50 / GB10)

## Purpose

Step-5's DSA attention backend is restricted to SM90 and SM103:

```
NotImplementedError: Step5 DSA requires SM90 or SM103; got DeviceCapability(major=12, minor=0).
```

The backend is built almost entirely on Triton and FlashInfer FA2, which already run on
SM12x. This PR enables SM120/SM121 with three small changes:

1. **Capability gate.** `Step5DSAAttentionBackend.supports_compute_capability`
   also accepts `(12, 0)` and `(12, 1)`.
2. **Indexer logits dispatch.** `Step5FlashInferSparse` chooses the logits kernel by
   arch:
   - SM100-family keeps `flashinfer.attn_scores.fp8_paged_mqa_logits`;
   - every other arch uses the existing Triton `paged_mqa_logits_sm90`, as Hopper already does.

   FlashInfer's `compute_paged_mqa_logits_schedule` is only called when the FlashInfer kernel is in use, because the Triton kernel ignores `schedule_meta`.
3. **Shared memory on SM12x.** SM12x has 101,376 B of opt-in shared memory per block, against 232,448 B on Hopper. On SM12x only, `_sparse_decode_split` launches with `num_stages=1`: each iteration gathers six 64×64 BF16 K/V tiles, about 48 KB per stage. `_direct_scores` launches with `num_stages=2`: a 128×256 FP8 key tile per iteration. Other archs keep Triton's default.

Optional and defensive: `layernorm.py` falls back to the PyTorch RMSNorm if the
CuTe-DSL kernel fails to compile or launch. It compiled and matched exactly on SM120.

## Test plan

Hardware: 8× RTX PRO 6000 Blackwell Server Edition (SM120, 96 GB, PCIe), driver
610.43.02, torch 2.13.0+cu130, Triton 3.7.1. Checkpoint: `stepfun-ai/Step-5-Preview-NVFP4`.

### Kernel tests against PyTorch references, single GPU

| Kernel | Max abs error |
|---|---|
| `paged_mqa_logits_sm90` | 3.8e-6 |
| `decode_summary_scores` | 1.9e-6 |
| `select_regions` (`per_row`) | identical to `torch.topk` |
| `sparse_decode_attention` (splits 1 / 32) | 1.6e-3 (BF16) |
| FlashInfer FA2 paged / ragged d192 | 3.9e-3 / 2.4e-3 |
| vllm FA2 paged d192 | 1.9e-3 |
| CuTe RMSNorm (+ FP32 residual) | 0 |

### End to end

Run with TP8, CUDA graphs, MTP k=3, `--block-size 64` and a 1 M max context:

- The server starts, and the DSA scored-prefill warmup passes.
- NVFP4 MoE uses `VLLM_CUTLASS`.
- KV cache holds 1,593,234 tokens.
- Prefill throughput is about 3.1–3.3 K tok/s from 12 K to 320 K prompt tokens, a PCIe-bound TP8 configuration.
- Generations are coherent, and token IDs match `tokenizer.json`.

## Notes

- I have no SM90 or SM100 hardware to run regressions on. The patch keeps the original
  code path for those archs: dispatch by `major == 10`, and stages only overridden when `major == 12`.
- The stage overrides are env-tunable (`STEP5_SM12X_DECODE_STAGES`,
  `STEP5_SM12X_SCORE_STAGES`). A cleaner upstream version could choose them from
  `shared_memory_per_block_optin`, or autotune them.
