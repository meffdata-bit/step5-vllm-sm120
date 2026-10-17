**Title:** Running Step-5-Preview-NVFP4 on 8× RTX PRO 6000 (SM120) with vLLM: working patch

Hi! In case it helps other people with consumer or workstation Blackwell cards:
the official `vllm/vllm-openai:step-5-preview` image stops on SM120 with
`Step5 DSA requires SM90 or SM103`.

The DSA path is almost all Triton + FlashInfer FA2, so it ports easily:

- allow SM12x in the backend's capability check;
- use the existing Triton indexer-logits kernel instead of the SM100-only FlashInfer one;
- lower Triton pipeline stages on SM12x, which has ~99 KB of shared memory per block.

Five patched Python files are bind-mounted over the image, with no rebuild.

With TP8, CUDA graphs, MTP and NVFP4 MoE on `VLLM_CUTLASS`:

- the KV cache holds ~1.59 M tokens, so a 1 M context fits;
- prefill runs at ~3.2 K tok/s on PCIe;
- generations are coherent;
- a smoke test checks every DSA kernel against PyTorch references.

Repo with patches, the smoke test, a docker-compose file and notes on what didn't work
(SGLang on SM120, Mooncake KV offload over TCP): <REPO_URL>

Unofficial, tested on a single machine. Feedback and reports from other SM120/SM121
setups (RTX 5090, GB10/DGX Spark) are welcome.
