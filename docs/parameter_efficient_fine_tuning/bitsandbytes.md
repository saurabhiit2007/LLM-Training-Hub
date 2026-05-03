# bitsandbytes

**bitsandbytes** is a lightweight CUDA extension by Tim Dettmers that provides:

- **8-bit and 4-bit quantized matrix multiplication** via `LLM.int8()` and NF4
- **8-bit optimizers** (`Adam8bit`, `PagedAdam8bit`) for 4× optimizer state memory reduction
- **Device-map-aware quantization** for loading large models across GPUs and CPU

It is the underlying library for QLoRA and is integrated into Hugging Face Transformers via the `BitsAndBytesConfig` class.

## Key APIs

```python
import bitsandbytes as bnb

# 8-bit Adam (4× less optimizer memory)
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-4)

# Paged 8-bit Adam (avoids OOM spikes via CPU paging)
optimizer = bnb.optim.PagedAdam8bit(model.parameters(), lr=2e-4)
```

For 4-bit model loading and QLoRA usage, see [QLoRA](qlora.md). For the block-wise quantization math behind 8-bit optimizers, see [Memory-Efficient Optimizers](../optimization/memory_efficient_optimizers.md).

## References

- [bitsandbytes GitHub](https://github.com/TimDettmers/bitsandbytes)
- [LLM.int8() Paper (2022)](https://arxiv.org/abs/2208.07339)
