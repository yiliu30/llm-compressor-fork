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