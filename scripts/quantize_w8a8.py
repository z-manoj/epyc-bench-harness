"""Quantize a HF safetensors model to INT8 W8A8 (compressed-tensors) for vLLM.

Round-to-nearest, no calibration data: per-channel symmetric INT8 weights, dynamic per-token INT8 activations.
RedHatAI's published W8A8 models use SmoothQuant + GPTQ; this is only for models they do not publish.

Run in the llm-compressor venv:
  ~/vllm-zen/llmc/bin/python scripts/quantize_w8a8.py <src_dir> <out_dir>
"""
import sys

import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

src, out = sys.argv[1], sys.argv[2]

model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(src)

# MoE router gates stay unquantized: they are tiny and INT8 routing changes expert selection.
recipe = QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head", "re:.*block_sparse_moe.gate$", "re:.*mlp.gate$"])
oneshot(model=model, recipe=recipe)

model.save_pretrained(out, save_compressed=True)
tokenizer.save_pretrained(out)
