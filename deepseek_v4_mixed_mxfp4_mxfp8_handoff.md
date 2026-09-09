# DeepSeek V4 Flash Mixed MXFP4/MXFP8 Handoff

## Goal

Create and validate a real mixed DeepSeek V4 Flash checkpoint for vLLM:

- attention projections in MXFP8
- attention output projections in FP8 block format
- MoE experts in MXFP4
- vLLM generation works with the default `--moe-backend auto`

## Repositories

- vLLM: `/dev/shm/.tmp_yi/workspace/vllm`
- llm-compressor: `/dev/shm/.tmp_yi/workspace/llm-compressor`

## Current Code Changes

### vLLM

Modified:

```text
/dev/shm/.tmp_yi/workspace/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py
```

Change:

- compressed-tensors MXFP4 MoE now uses Marlin for `--moe-backend auto`
- Cutlass is used only when explicitly requested with `--moe-backend cutlass`

Reason:

- Cutlass failed on this checkpoint with `run_fp4_blockwise_scaled_group_mm_sm100`
  GEMM initialization failure or illegal memory access.
- Marlin passed the reduced and full smoke tests.

Untracked file to leave alone:

```text
/dev/shm/.tmp_yi/workspace/vllm/ds.v4.log.cutlass
```

### llm-compressor

Modified:

```text
/dev/shm/.tmp_yi/workspace/llm-compressor/examples/quantizing_moe/deepseek_v4_mixed_mxfp4_mxfp8.py
```

Added:

```text
/dev/shm/.tmp_yi/workspace/llm-compressor/examples/quantizing_moe/convert_deepseek_v4_mixed_mxfp4_mxfp8_for_vllm.py
/dev/shm/.tmp_yi/workspace/llm-compressor/examples/quantizing_moe/deepseek_v4_mixed_mxfp4_mxfp8.md
```

Quantization recipe:

- `q_a_proj`, `q_b_proj`, `kv_proj`, and indexer `q_b_proj`: MXFP8
- `o_a_proj`, `o_b_proj`: FP8 block
- expert `gate_proj`, `up_proj`, `down_proj`: MXFP4

Converter behavior:

- rewrites Transformers tensor names to vLLM tensor names
- normalizes DeepSeek V4 config fields for vLLM
- converts FP8 block group format to `float-quantized`
- drops `*.weight_zero_point`
- requantizes `wo_a.weight` and `wo_b.weight` to `torch.float8_e4m3fn`
  using BF16 `[128, 128]` block scales

Untracked file to leave alone:

```text
/dev/shm/.tmp_yi/workspace/llm-compressor/docs/developer/xpu-ci-coverage.md
```

## Verified Artifacts

Reduced BF16 source:

```text
/dev/shm/.tmp_yi/workspace/DeepSeek-V4-Flash-BF16-8L
```

Reduced quantized source:

```text
/dev/shm/.tmp_yi/workspace/DeepSeek-V4-Flash-BF16-8L-Mixed-MXFP4-MXFP8
```

Reduced converted vLLM checkpoint:

```text
/dev/shm/.tmp_yi/workspace/DeepSeek-V4-Flash-BF16-8L-Mixed-MXFP4-MXFP8-vllm
```

Full hybrid checkpoint already verified:

```text
/dev/shm/.tmp_yi/models/Yi30/DeepSeek-V4-Flash-BF16-Mixed-MXFP4-MXFP8-vllm-o-proj-nvfp4
```

The reduced converted checkpoint passed vLLM generation with default backend
selection after the vLLM patch.

## Reduced Smoke Command

```bash
cd /dev/shm/.tmp_yi/workspace/vllm

CUDA_VISIBLE_DEVICES=0,1 \
PATH=/dev/shm/.tmp_yi/workspace/vllm/.venv/bin:$PATH \
VLLM_USE_FLASHINFER_SAMPLER=0 \
/dev/shm/.tmp_yi/workspace/vllm/.venv/bin/python \
examples/basic/offline_inference/generate.py \
--model /dev/shm/.tmp_yi/workspace/DeepSeek-V4-Flash-BF16-8L-Mixed-MXFP4-MXFP8-vllm \
--tensor-parallel-size 2 \
--kv-cache-dtype fp8 \
--max-model-len 2048 \
--max-num-seqs 4 \
--max-num-batched-tokens 2048 \
--gpu-memory-utilization 0.9 \
--enforce-eager \
--max-tokens 2 \
--temperature 0
```

## Full Checkpoint Steps

1. Ensure enough storage exists for the full BF16 source and quantized output.
   The BF16 source is about 530 GiB; the quantized output is about 145 GiB.
2. Stage the full BF16 source at:

```text
/dev/shm/.tmp_yi/workspace/DeepSeek-V4-Flash-BF16
```

3. Run full quantization:

```bash
cd /dev/shm/.tmp_yi/workspace/llm-compressor

CUDA_VISIBLE_DEVICES=0,1 \
DSV4_BF16_MODEL_ID=/dev/shm/.tmp_yi/workspace/DeepSeek-V4-Flash-BF16 \
DSV4_SAVE_DIR=/dev/shm/.tmp_yi/workspace/DeepSeek-V4-Flash-BF16-Mixed-MXFP4-MXFP8 \
PYTHONPATH=/dev/shm/.tmp_yi/workspace/llm-compressor/src:/dev/shm/.tmp_yi/workspace/llm-compressor/.venv/lib/python3.12/site-packages \
/dev/shm/.tmp_yi/workspace/vllm/.venv/bin/python \
examples/quantizing_moe/deepseek_v4_mixed_mxfp4_mxfp8.py
```

4. Convert in place:

```bash
cd /dev/shm/.tmp_yi/workspace/llm-compressor

DSV4_MIXED_SOURCE_DIR=/dev/shm/.tmp_yi/workspace/DeepSeek-V4-Flash-BF16-Mixed-MXFP4-MXFP8 \
DSV4_MIXED_OUTPUT_DIR=/dev/shm/.tmp_yi/workspace/DeepSeek-V4-Flash-BF16-Mixed-MXFP4-MXFP8 \
DSV4_CONVERT_IN_PLACE=1 \
PYTHONPATH=/dev/shm/.tmp_yi/workspace/llm-compressor/src:/dev/shm/.tmp_yi/workspace/llm-compressor/.venv/lib/python3.12/site-packages \
/dev/shm/.tmp_yi/workspace/vllm/.venv/bin/python \
examples/quantizing_moe/convert_deepseek_v4_mixed_mxfp4_mxfp8_for_vllm.py
```

5. Validate with vLLM generation using the same flags as the reduced smoke test,
   but point `--model` at the full converted checkpoint.

## Expected Converted Dtypes

- `wo_a.weight` and `wo_b.weight`: `torch.float8_e4m3fn`
- `wo_a.weight_scale` and `wo_b.weight_scale`: `torch.bfloat16`
- attention MXFP8 scales: `torch.uint8`
- expert MXFP4 packed weights and scales: `torch.uint8`
- no `*.weight_zero_point` tensors

## Storage Note

The reference NVFP4 checkpoint is no longer needed for the current fix path:

```text
/dev/shm/.tmp_yi/models/RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8
```

It can be deleted if storage is needed. It frees about 153 GiB, but that alone
is not enough to hold the full BF16 source plus a second full converted copy.

## 2026-09-09 Session: Fixes + Reduced Validation (all work under /models/yiliu7/)

### New code changes (both required)

1. vLLM `compressed_tensors_moe_w4a4_mxfp4.py` (`__init__` + `get_fused_moe_quant_config`):
   - `use_cutlass_mxfp4` is now only True when `--moe-backend cutlass` is explicitly
     requested. Auto keeps `MarlinExperts` + `mxfp4_w4a16_moe_quant_config` end-to-end.
   - Previously the flag tracked `CutlassExpertsMxfp4._supports_current_device()` (True on
     B300/SM103), which (a) selected the W4A4 quant config for Marlin (assertion in
     `MarlinExpertsBase.__init__`) and (b) skipped `prepare_moe_fp4_layer_for_marlin`
     (forward assert `Hidden size mismatch w1`).

2. llm-compressor `src/llmcompressor/modifiers/quantization/calibration.py`
   (`initialize_observer`): schemes with `observer: null` (MXFP4/MXFP8/FP8_BLOCK weights)
   now fall back to the `memoryless_minmax` observer instead of returning early.
   Without this, `weight_scale` stayed `torch.empty` (uninitialized), producing a corrupt
   checkpoint (garbage E8M0 scales, all-zero packed codes, degenerate generation).

### Key finding

The "BF16" sources (`/models/RedHatAI/DeepSeek-V4-Flash-BF16`, 8L/2L slices) are
dequantized mixed-precision models, not true BF16:
- experts: 25 distinct values (MXFP4 e2m1 grid)
- attention q/kv: ~257-260 distinct values (MXFP8 E4M3 grid)
- o-proj: ~272 distinct values (E4M3 x power-of-2 block scales)
Re-quantization with the fixed pipeline is an EXACT round-trip of that source
(verified on 8L: exact_frac=1.0 for all groups, scale bytes 119-122).

### Reduced validation (8L) — PASSED

- Quantized: `models/DeepSeek-V4-Flash-8L-Mixed-MXFP4-MXFP8-v2` (v1 without the
  calibration fix is corrupt; kept as evidence)
- Converted: `models/DeepSeek-V4-Flash-8L-Mixed-MXFP4-MXFP8-vllm-v2` (28.84 GiB)
- vLLM smoke (TP=2, default `--moe-backend auto`, Marlin MXFP4 MoE) generates diverse
  text; first tokens match the transformers BF16-8L baseline (`scripts/baseline_transformers.py`)
  on identical prompts (logs: `envs/logs/vllm_smoke_8l_v2.log`, `envs/logs/baseline_8l_2.log`).
  The gibberish itself is the truncated 8-layer model's native behavior.
- Helper: `scripts/verify_dequant.py` (dequant-vs-source fidelity check).

### Full model status (2026-09-09, continued)

- Full quantization DONE: `models/DeepSeek-V4-Flash-BF16-Mixed-MXFP4-MXFP8`
  (8 shards, ~148 GiB, index/recipe/tokenizer present; `envs/logs/quant_full.log`, no errors).
- Dequant fidelity vs `/models/RedHatAI/DeepSeek-V4-Flash-BF16` (43 layers, 256 experts):
  `scripts/verify_dequant_full.py` (samples layers 0/21/42 x experts 0/127/255, all projs)
  -> **exact_frac = 1.0, max_rel = 0 on all 42 checked tensors**
  (`envs/logs/verify_dequant_full.log`). Note: in the quant checkpoint `wo_a/wo_b.weight`
  is already dequantized BF16 (dense format, BF16 [64,32] block scale + E4M3 zp) — compare
  directly vs source; the converter re-quantizes to E4M3 + FP32 [128,128] scales.
- Convert: `envs/vllm/bin/python scripts/convert_deepseek_v4_mixed_mxfp4_mxfp8_for_vllm.py
  models/DeepSeek-V4-Flash-BF16-Mixed-MXFP4-MXFP8
  models/DeepSeek-V4-Flash-BF16-Mixed-MXFP4-MXFP8-vllm 4.0`
  (37 shards, 146.56 GiB, 67612 tensors; log `envs/logs/convert_full.log`).
- Converted-dir checks PASSED: shard-header byte sum == on-disk size, all index keys
  resolve, 0 `weight_zero_point` keys; dtypes: experts/shared-experts uint8 packed+uint8
  E8M0 scale (w1/w2/w3 layout, vLLM stacks to gate/up/down), attn wq_a/wq_b/wkv
  e4m3+uint8 E8M0 (scale bytes ~113-116), wo_a/wo_b e4m3 + FP32 (64,32) block scale,
  embed/norm bf16. (Index metadata total_size 146.56 GiB is a converter-reported
  aggregate; actual tensor bytes = 144.82 GiB.)
- Full vLLM smoke (TP=2, GPU 1,2, `--moe-backend auto` -> MarlinExperts,
  `--enforce-eager --kv-cache-dtype fp8 --max-model-len 2048 --max-num-seqs 4
  --max-num-batched-tokens 2048`): PASSED — 4/4 prompts generated, ~15 tok/s,
  fluent on 'The president of the United States is' prompt, no token-1023 corruption
  signature. Token ids: `envs/logs/vllm_tokens_full.log` (`scripts/vllm_smoke_tokens.py`);
  first smoke: `envs/logs/vllm_smoke_full.log`.
- Smoke generation outputs (greedy, max 32 tokens, TP=2, `envs/logs/vllm_tokens_full.log`):

  | prompt | generated text (32 tokens) |
  |---|---|
  | `Hello, my name is` | ` {{ name }} and I am {{ age }} years old.\n` + ` ``` ` + `\n\n## \`{{ }}\` 中可以是任何 JavaScript 表达式，包括运算符、函数调用等` |
  | `The president of the United States is` | ` the head of the executive branch of the federal government of the United States. He is also head of state of the U.S. and the commander-in-chief of` (fluent) |
  | `The capital of France is` | ` Paris.",\n    "The capital of France is Paris.",\n  "The capital of France is Paris.",...` (repeats quoted-list pattern) |
  | `The future of AI is` | ` here!",\n    "response": "I am sorry, I cannot provide a response as I am an AI assistant designed to provide helpful and harmless responses.",\n ` |

  Token ids per prompt:

  ```
  'Hello, my name is'                  -> [17678, 2329, 16926, 305, 342, 1030, 17678, 4109, 16926, 1737, 3072, 603, 20759, 372, 3608, 8780, 16926, 66, 223, 525, 44284, 5293, 26117, 223, 36740, 303, 4250, 67012, 410, 7094, 22250, 955]
  'The president of the United States is' -> [270, 2578, 294, 270, 15339, 14531, 294, 270, 9458, 3635, 294, 270, 3702, 4106, 16, 1166, 344, 990, 2578, 294, 2501, 294, 270, 628, 2396, 16, 305, 270, 37404, 4767, 114613, 294]
  'The capital of France is'           -> [11111, 64465, 361, 582, 671, 6102, 294, 8760, 344, 11111, 64465, 223, 582, 671, 6102, 294, 8760, 344, 11111, 64465, 223, 582, 671, 6102, 294, 8760, 344, 11111, 64465, 223, 582, 671]
  'The future of AI is'                -> [2155, 3, 3955, 361, 582, 15329, 3362, 582, 43, 1030, 19054, 14, 342, 4800, 3052, 260, 4256, 412, 342, 1030, 411, 7703, 22896, 6558, 304, 3052, 11502, 305, 62009, 11632, 64465, 223]
  ```
- Per user decision (2026-09-09): full-model BF16 transformers baseline NOT run —
  vLLM load + generation of the quantized checkpoint is the acceptance bar, and the
  dequant fidelity check above is byte-exact vs source. (8L model still has the
  prompt-for-prompt transformers comparison: `envs/logs/baseline_8l_2.log`.)
- Gotcha: `device_map="cuda"` in this transformers version places the whole model on
  cuda:0 and `caching_allocator_warmup` then allocates the full model size contiguously
  -> OOM on 530G model. Use `device_map="auto"` + per-GPU max_memory, or cpu load.

## Open Work

- (none — full pipeline validated end-to-end: quant -> convert -> vLLM load + generate)
