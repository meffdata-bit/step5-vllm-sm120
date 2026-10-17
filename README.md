# Step-5-Preview on RTX PRO 6000 Blackwell (SM120) with vLLM

Unofficial patch set that lets **`stepfun-ai/Step-5-Preview-NVFP4`** run on
**8× NVIDIA RTX PRO 6000 Blackwell (SM120, 96 GB, PCIe, no NVLink)** with the
official `vllm/vllm-openai:step-5-preview` image.

Out of the box, the image refuses to build the model on SM120:

```
NotImplementedError: Step5 DSA requires SM90 or SM103; got DeviceCapability(major=12, minor=0).
```

These patches fix that by bind-mounting five modified Python files over the
image's originals. Nothing is rebuilt, and no new kernels are compiled ahead of time.

> Not affiliated with StepFun or the vLLM project. Tested on one machine, so
> treat it as a starting point. If you hit a different error, please open an issue.

## Why it fails, and what the patch changes

The vLLM implementation of Step-5's sparse attention (DSA) is almost entirely
**Triton + FlashInfer FA2**, which already run on SM120. Only three things tie it to
Hopper/B300:

| # | Problem | Fix | File |
|---|---|---|---|
| 1 | Hard capability gate: only SM90 and SM103 accepted | Also accept SM120 / SM121 | `sparse_attention.py` |
| 2 | The FP8 indexer-logits kernel used off-Hopper (`flashinfer.attn_scores.fp8_paged_mqa_logits`) targets SM100-class GPUs | Use the existing portable Triton kernel (`paged_mqa_logits_sm90`) on every non-SM100 arch; skip FlashInfer's schedule helper there | `flashinfer_sparse.py` |
| 3 | Decode kernels tuned for 227 KB shared memory per block; SM12x has ~99 KB | Lower Triton `num_stages` **only on SM12x** (decode attention: 1, decode indexer scores: 2; env-overridable) | `ops/sparse_decode_attention.py`, `ops/sparse_indexer_scores.py` |
| — | CuTe-DSL RMSNorm might not compile for a new arch | Safety net: fall back to the existing PyTorch path on failure (it compiled fine on SM120 in our tests) | `layernorm.py` |

On other GPUs the behaviour is unchanged. Every change is marked `[SM120 port]`.
The full diff against the image is in [`step5-sm120.patch`](step5-sm120.patch).

## Tested environment

| | |
|---|---|
| GPUs | 8× RTX PRO 6000 Blackwell **Server Edition**, 96 GB, PCIe Gen5 x16, compute capability 12.0 |
| Host | Dual-socket AMD, 2 GPUs per PCIe switch, Ubuntu, kernel 7.0, driver 610.43.02 |
| Image | `vllm/vllm-openai:step-5-preview` built 2026-10-14 (vLLM `0.1.dev22912+g8224bb82f`, torch 2.13.0+cu130, Triton 3.7.1) |
| Checkpoint | `stepfun-ai/Step-5-Preview-NVFP4` (394 GB, ModelOpt NVFP4) |

The patched files replace whole files from **this specific image build**. If the
`step-5-preview` tag moves to a newer build, re-run `./run_smoke.sh` before
serving, and compare the files with `step5-sm120.patch`. Pinning the image by digest
is recommended (`docker image inspect --format '{{index .RepoDigests 0}}' vllm/vllm-openai:step-5-preview`).

## Usage

```bash
git clone <this repo> && cd step5-vllm-sm120

# 1) Kernel smoke test on one GPU (~2 min, no weights needed)
./run_smoke.sh

# 2) Serve (expects the checkpoint in ~/.cache/huggingface, or set MODELS_DIR)
docker compose up -d
docker compose logs -f
```

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"step5","messages":[{"role":"user","content":"Hello!"}],"max_tokens":512}'
```

If the checkpoint is not local yet, remove `HF_HUB_OFFLINE=1` from the compose, or
download it first with `hf download stepfun-ai/Step-5-Preview-NVFP4`.

## Results

### Kernel smoke test (`smoke_sm120.py`)

Each test JIT-compiles one kernel of the DSA path on SM120 and compares it with a
plain PyTorch reference:

| Test | Max abs error |
|---|---|
| RMSNorm (CuTe DSL), with/without FP32 residual | 0 |
| Indexer logits, prefill (Triton FP8) | 3.8e-6 |
| Indexer logits, decode (Triton FP8) | ≤ 1.9e-6 |
| Region top-k (`per_row`) | selection identical to `torch.topk` |
| Sparse decode attention (Triton), 1 and 32 splits | ≤ 1.6e-3 (BF16) |
| FlashInfer FA2 paged / ragged, head_dim 192 | 3.9e-3 / 2.4e-3 (BF16) |
| vLLM FlashAttention-2 paged, head_dim 192 | 1.9e-3 (BF16) |

### Full model

- **Startup:** NVFP4 MoE runs on `VLLM_CUTLASS` (native SM120 FP4 kernels). The BF16 MTP experts use Triton MoE.
- **Weights:** ~48.5 GB per GPU.
- **KV cache:** **1,593,234 tokens** with `--gpu-memory-utilization 0.93 --block-size 64`. That is 1.43 M with the default block size 16. A 1 M-token request fits (1.52×).
- **Working features:** CUDA graphs and MTP speculative decoding (3 tokens), plus the reasoning and tool-call parsers.
- **Tokenization:** vLLM's token IDs are identical to raw `tokenizer.json` on Spanish text, code, digits and whitespace. The `transformers` "incorrect regex pattern" warning at startup is a false positive for this model.
- **Prefill throughput:** close to linear in context length, thanks to the sparse attention:

  | Prompt tokens | Time to first token | Throughput |
  |---|---|---|
  | 12,385 | 3.8 s | 3,272 tok/s |
  | 79,316 | 24.6 s | 3,219 tok/s |
  | 320,139 | 104.8 s | 3,054 tok/s |

  Repeating a cached prefix: < 1 s.

## Performance notes for PCIe-only, dual-socket boxes

- Prefill is bound by TP8 all-reduce traffic. On our board, NCCL all-reduce
  reached only ~17–18 GB/s bus bandwidth, for 2, 4 and 8 GPUs alike. Raw P2P copies
  run at ~48 GB/s. The limit is that **each PCIe switch has a single x16 uplink shared by two GPUs**.
- **`NCCL_P2P_LEVEL=SYS`** cut small (≤ 1 MB) all-reduce latency by 25–30%, which helps decode. It made no difference for large messages.
- These changes did **not** help on our machine:
  - more NCCL channels;
  - `NCCL_P2P_DISABLE`;
  - disabling PCIe ACS (P2P was already direct, `P2P/CUMEM`).
- TP4×PP2 aligned to sockets is unlikely to pay off when intra-socket all-reduce is as slow as cross-socket all-reduce, and it would slow down single-user decode.

## Things we tried that did not work (to save you time)

- **SGLang (`lmsysorg/sglang:step-5-preview`, built 2026-10-15)** gets further than stock vLLM on SM120: it loads weights and allocates the KV cache. It then needs `--model-path` set to the local snapshot, `--page-size 64` and `--tp 8 --ep-size 8`. It finally fails at warmup in a CuTe Q/K-norm kernel limited to Hopper and datacenter Blackwell (`Step5 Q/K norm port supports Hopper and Blackwell`).
- **KV offload to host RAM with `MooncakeStoreConnector`**, the only KV connector vLLM allows for Step-5 DSA, over TCP on a single node:
  - Puts first failed for 7 of 8 keys with `-800` (cross-process TCP lane queue overflow, from the many small slices of the `LBNHC` layout). Bigger `MC_TCP_*` queues and `--block-size 64` stopped the put failures.
  - Saving still ran at only ~45 MB/s, and loads failed with `TRANSFER_FAIL` / `LEASE_EXPIRED`.
  - With the default `kv_load_failure_policy=fail`, **requests then error out**.
  - Not usable without RDMA.

## Tuning knobs added by the patch

| Env var | Default | Effect |
|---|---|---|
| `STEP5_SM12X_DECODE_STAGES` | `1` | Triton `num_stages` for sparse decode attention on SM12x |
| `STEP5_SM12X_SCORE_STAGES` | `2` | Triton `num_stages` for decode indexer scores on SM12x |
| `STEP5_TORCH_RMSNORM` | `0` | `1` forces the PyTorch RMSNorm instead of CuTe DSL |

The stage counts are conservative. Raising them may speed up decode, but if a value
does not fit in shared memory, Triton fails with `OutOfResources`.

## Status / upstream

The proper fix belongs in vLLM: the capability gate, the logits dispatch by arch,
and shared-memory-aware launch configs. See [`UPSTREAM_PR.md`](UPSTREAM_PR.md) for a
ready-to-submit description. Once upstream supports SM120, drop the volume mounts.

## License

The patched files are derived from vLLM and remain under the **Apache License 2.0**
([LICENSE](LICENSE)). Modifications are marked inline with `[SM120 port]`. The
helper scripts and docs in this repo are released under the same license. The
Step-5 model weights are under the StepFun Model License, which is not covered here.
