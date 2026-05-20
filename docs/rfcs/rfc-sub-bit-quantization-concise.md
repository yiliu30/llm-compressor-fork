# RFC: Sub-Bit Quantization Support With Humming

Author: Yi Liu  
Date: 2026-05-20  
Status: Draft

## Summary

Enable weight-only sub-bit quantization for 3/5/6/7-bit models across:

llm-compressor -> compressed-tensors -> vLLM -> humming

The target is to let users compress a model with sub-bit weight quantization and serve it in vLLM through the standard compressed-tensors path, with humming selected underneath as the runtime backend for sub-bit execution.

## Target

The target of this RFC is:

- llm-compressor can produce compressed-tensors checkpoints for 3/5/6/7-bit weight-only quantization.
- compressed-tensors can serialize those checkpoints correctly.
- humming can consume the resulting checkpoint format correctly, without repack if possible.
- vLLM can run those models through the default compressed-tensors path without requiring an explicit `--quantization humming` override.

## Scope

In scope:

- Weight-only INT 3/5/6/7-bit support.
- Target modifiers: `GPTQModifier`, `AutoRoundModifier`, and `QuantizationModifier` for weight-only sub-bit flows.
- End-to-end support through the default compressed-tensors inference path in vLLM for sub-bit models.
- High-level repo changes and validation requirements.



## Packing Solutions Considered

For odd bit-widths, the main technical choice is the checkpoint packing format.

### Option A: Keep Current compressed-tensors Packing

- Keep the existing compressed-tensors padded packing for 3/5/6/7-bit weights.
- Make humming and the vLLM compressed-tensors path understand and convert that layout correctly at load time.
- This keeps the current serialization behavior but pushes more complexity into the runtime path.

### Option B: Add Dense Odd-Bit Packing

- Add a dense odd-bit packing path in compressed-tensors for 3/5/6/7-bit weights.
- Align the stored layout more closely with what humming expects.
- Keep the serving path simpler because less odd-bit-specific conversion is required at runtime.

### Preferred Direction

- Prefer Option B for newly generated 3/5/6/7-bit checkpoints.
- Keep 4-bit and 8-bit behavior unchanged.
- Treat the packing decision as part of the compressed-tensors work, with validation in humming and vLLM.

## High-Level Changes By Repo

### compressed-tensors

Required changes:

- Remove the current 4-bit / 8-bit restriction for pack-quantized weight-only compression.
- Add standard sub-bit preset schemes such as `W3A16`, `W5A16`, `W6A16`, and `W7A16`.


### llm-compressor

Required changes:

- Validate that existing quantization flows can emit sub-bit compressed-tensors checkpoints.
- Add or update examples and recipe guidance for sub-bit usage.

### humming

Required changes:

- Validate that compressed-tensors sub-bit checkpoints are interpreted correctly.
- Fix schema-loading or conversion logic if the current compressed-tensors path is only safe for 4-bit / 8-bit assumptions.

### vLLM

Required changes:

- Extend the compressed-tensors path so sub-bit checkpoints select a humming-backed kernel by default.
- Verify end-to-end loading and inference without requiring `--quantization humming`.
- Keep the solution scoped to sub-bit compressed-tensors serving rather than a broader kernel-framework redesign.

## User Experience Target

Compression:

```python
recipe = GPTQModifier(scheme="W3A16")
recipe = AutoRoundModifier(scheme="W5A16")
```

Inference:

```bash
vllm serve ./my-model
```

## Validation Requirements

- Validate at least 3-bit and 5-bit end to end.
- Confirm checkpoint save/load works without shape or packing errors.
- Run generation successfully in vLLM through the default compressed-tensors path.
- Compare behavior against an existing 4-bit baseline.
- Cover edge cases that matter for packed formats, including odd-bit layout assumptions and metadata correctness.

## Success Criteria

This RFC is successful when:

- A model can be compressed to 3/5/6/7-bit with llm-compressor and compressed-tensors.
- The produced checkpoint can be loaded by humming.
- The model runs in vLLM through the normal compressed-tensors path.
- The solution is documented at a high level and validated end to end.

## Non-Goals For This RFC

- Reordering kernel priority among Marlin, Machete, Exllama, and humming.
- Introducing a broader mixed-precision kernel redesign.