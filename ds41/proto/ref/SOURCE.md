# Source of these files

Copied unmodified from https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
(`inference/` and `encoding/encoding.py`), branch main at commit
2cba9e42aa026125f3ed06c6d98c1db82f7ca027, on 2026-10-05. License: MIT (see LICENSE).

`proto/ds41_proto.py` imports `model.py` and patches only: routed experts (EXL3 3bpw from
coolbho3k), the Engram table lookup (mmap from SSD), the output head storage (bf16), and the
`kernel` module (TileLang or the PyTorch equivalents in `proto/torch_kernels.py`).
