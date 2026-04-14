Goal: Add RCEIL support to LLMC.

Context: There are several rounding methods available, such as FLOOR, CEIL, and RCEIL. Please refer to torchao's ScaleCalculationMode for details.

Currently, compressed-tensor already supports the FLOOR option. We want to extend this to include additional rounding modes, such as RCEIL.

### Targets:
- Extend the quantization configuration to allow users to specify different rounding methods.
- Provide an end-to-end example with reasonable output.

### References:
- torchao ScaleCalculationMode: /home/yiliu7/workspace/ao/torchao/prototype/mx_formats/config.py
- compressed-tensor: /home/yiliu7/workspace/compressed-tensors/src/compressed_tensors/compressors/mx_utils.py
- compressed-tensor is a standalone repository that provides quantization primitives for llm-compressor.
- End-to-end example: /home/yiliu7/workspace/llm-compressor/inc/experimental/mxfp4/qwen_mxfp4.py

### Progress
- Implemented the core feature across both repositories.
- In `compressed-tensors`:
  - Added `ScaleCalculationMode` with `floor` and `rceil`.
  - Added `scale_calculation_mode` to `QuantizationArgs`, defaulting to `floor`.
  - Threaded the new field through MX scale generation and MX scale compression for MXFP4 and MXFP8.
  - Added unit coverage for quant arg validation/serialization, MXFP4/MXFP8 scale generation, and compressor behavior.
- In `llm-compressor`:
  - Added a quantization config resolution test proving `scale_calculation_mode="rceil"` passes through correctly.
  - Restored a Qwen MXFP4 experimental example at `experimental/mxfp4/qwen_mxfp4.py`.
  - Added a short standalone rounding explainer at `experimental/mxfp4/rounding_modes_example.py`.
  - Updated `experimental/mxfp4/README.md` to point to the new examples.

### Verification
- `compressed-tensors` targeted tests passed: `29 passed`
- `llm-compressor` targeted quantization test passed: `36 passed`
- `ruff check` passed on all touched Python files
- `python -m compileall` passed for the new experimental scripts
- `python experimental/mxfp4/rounding_modes_example.py` ran successfully and showed the expected `floor` vs `rceil` difference

### Remaining Gap
- The full end-to-end Qwen example was added but not executed in this task because it requires downloading and quantizing a large model.
