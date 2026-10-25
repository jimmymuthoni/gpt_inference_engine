#### GPT Inference Engine.

A from-scratch C++ inference engine that runs GPT-2 (124M) on the CPU — no PyTorch, no GPU, no ML libraries at runtime.

You give it a prompt, and it generates text one token at a time, streaming it to the terminal, then reports generation speed (tokens/sec) and perplexity.

```
Prompt: "what is the weather today?"
Generated: ...

temperature sampling speed: XX.XX tokens/sec
the perplexity score: XX.XX
```

---

#### What is an inference engine?

A trained model is just a large file of numbers (weights). An inference engine is the runtime that loads those weights and executes the model's math to produce output. Training happens elsewhere; the engine's job is to run the finished model fast and memory-efficiently on the available hardware.

Well-known examples are `llama.cpp` and `llm.c`. This project is a small, readable implementation of the same idea, built to understand what happens underneath frameworks like PyTorch.

It focuses on:

- Latency / throughput — SIMD dot products and a KV cache so past tokens are never recomputed.
- Memory efficiency — weights are memory-mapped (`mmap`), not copied; an int8 quantization path is in progress.
- Hardware utilization — ARM NEON on ARM CPUs, AVX2 + FMA on x86-64, with a portable scalar fallback.

---

#### Architecture

```
HuggingFace "gpt2" (model.safetensors)
        │
        │  scripts/convert_weight.py      (float32)
        │  scripts/absmax.py              (int8, experimental)
        ▼
model/model.bin + model_index.json         raw tensors, 32-byte aligned, matrices transposed
        │
        │  src/load_gpt2.cpp              mmap + fixed offsets
        ▼
┌──────────────────────── C++ runtime ────────────────────────┐
│  Tokenizer (bpe.cpp)                                        │
│      ↓  token ids                                           │
│  Embedding  wte[token] + wpe[pos]                           │
│      ↓                                                      │
│  12 × Transformer block (model.cpp)                         │
│      x += Attention(LayerNorm1(x))   ← reads/writes KV cache│
│      x += MLP(LayerNorm2(x))                                │
│      ↓                                                      │
│  Final LayerNorm → logits (dot with wte, tied weights)      │
│      ↓                                                      │
│  Sampling (ops.cpp): greedy or temperature                  │
│      ↓                                                      │
│  Decode token → print → feed back in                        │
└─────────────────────────────────────────────────────────────┘
```

#### Model configuration (GPT-2 small)

| Parameter        | Value  |
|------------------|--------|
| Layers           | 12     |
| Attention heads  | 12     |
| Embedding dim    | 768    |
| Head dim         | 64     |
| MLP hidden dim   | 3072   |
| Context length   | 1024   |
| Vocabulary size  | 50257  |

These values are hardcoded in `src/load_gpt2.cpp` and `src/model.cpp`.

---

#### Project structure

```
.
├── main.cpp                   Entry point: generation loop, perplexity, timing
├── include/
│   ├── tensor.h               Tensor<N> (float) and Tensor_Quant<N> (int8) views
│   ├── model.h                Model, TransformerBlock, attention, MLP, LayerNorm
│   ├── model_quant.h          int8 versions of the above
│   ├── bpe.h                  Tokenizer encoder/decoder
│   └── ops.h                  Dot products, sampling
├── src/
│   ├── load_gpt2.cpp          mmaps model/model.bin and points tensors into it
│   ├── load_gpt2_quant.cpp    Same for the int8 model (model/model_absmax.bin)
│   ├── model.cpp              Float32 forward pass
│   ├── model_quant.cpp        int8 forward pass (work in progress)
│   ├── ops.cpp                SIMD kernels (NEON / AVX2 / scalar) and samplers
│   └── bpe.cpp                Vocabulary loading, encoding, decoding
├── scripts/
│   ├── convert_weight.py      safetensors → model.bin (float32)
│   ├── absmax.py              safetensors → int8 absmax-quantized binary
│   └── verify_conversion.py   Reads model.bin back to check shapes
├── CMakeLists.txt
├── setup.sh                   Clean build + run
├── pyproject.toml / uv.lock   Python dependencies
└── requirements.txt
```

---

#### How it works

### 1. Weight conversion (Python)

`scripts/convert_weight.py` downloads GPT-2 from HuggingFace and rewrites it into a format the C++ code can use directly:

1.Parses safetensors by hand. The file is `[8-byte header length][JSON metadata][raw tensor bytes]`. The script memory-maps it and uses each tensor's `data_offsets` to locate its bytes.
2. Writes tensors in a fixed order: `wte`, `wpe`, `ln_f`, then each layer's LayerNorm, attention and MLP parameters.
3. Transposes weight matrices. GPT-2 stores linear layers as `[in, out]`. They are written as `[out, in]` so every output neuron's weights are one contiguous row — a matrix-vector product becomes a series of contiguous dot products, which is ideal for SIMD.
4. Pads every tensor to 32 bytes for aligned SIMD loads.
5. Writes `model_index.json` with each tensor's offset, size, shape and dtype.

#### 2. Loading (C++)

`src/load_gpt2.cpp` memory-maps `model/model.bin`. Nothing is copied: each `Tensor` is a non-owning view pointing into the mapped file at a fixed offset (taken from `model_index.json`). The OS pages weights in on demand, so startup is near-instant.

> If you change the conversion order or padding, the offsets in `load_gpt2.cpp` must be regenerated from the new `model_index.json`.

#### 3. Tensors

`Tensor<N>` (`include/tensor.h`) is a minimal N-dimensional view: a shape, a data pointer, and an optional owned allocation (32-byte aligned). Indexing is bounds-checked, and `slice(i)` returns a sub-tensor without copying — used to hand each layer its slice of the KV cache.

`Tensor_Quant<N>` is the int8 equivalent, carrying a `scale` and dequantization helpers.

#### 4. Tokenization

- Decoder: loads `model/vocab.bin`, a sequence of `[1-byte length][token bytes]` records; the token id is its position in the file.
- Encoder: greedy longest-match against the vocabulary. This approximates GPT-2's BPE but does not apply merge rules, so tokenization can differ from HuggingFace's for some inputs.

#### 5. Forward pass

The engine processes one token per step:

1. Embedding: `x = wte[token] + wpe[position]`.
2. 12 transformer blocks, pre-LayerNorm with residual connections:
   - LayerNorm: mean and variance in a single pass (`var = E[x²] − E[x]²`), then scale and shift.
   - Causal self-attention: the fused `c_attn` matrix produces Q, K and V. K and V are written straight into the KV cache at the current position. Each of the 12 heads scores Q against all cached keys (scaled by `1/√64`), applies a numerically stable softmax, and takes a weighted sum of cached values. Causality is automatic because only past positions exist in the cache. Output goes through `c_proj` and is added to the residual.
   - MLP: 768 → 3072 → GELU (sigmoid approximation `x·σ(1.702x)`) → 768, added to the residual.
3. LM head: final LayerNorm, then a dot product with each row of `wte` (GPT-2 ties input and output embeddings) to produce 50257 logits.

KV cache layout: `Tensor<3>(12, 1024, 1536)` → `[layer][position][K(768) | V(768)]`.

#### 6. Kernels and sampling (`src/ops.cpp`)

- `sdot_simd` — the hot loop of the whole engine. Selected at compile time:
  - ARM NEON: 4 floats per step.
  - x86 AVX2 + FMA: 8 floats per step.
  - Scalar fallback for any other CPU.
- `sample_greedy` — picks the highest logit.
- `temperature_sampling` — divides logits by the temperature, applies softmax, and samples from the resulting distribution.

#### 7. Generation loop (`main.cpp`)

1. Encode the prompt.
2. Prefill: feed prompt tokens through the model to fill the KV cache, without sampling.
3. Decode: after the last prompt token, compute logits, sample a token, print it immediately, and feed it back in as the next input.
4. Report tokens/sec.
5. Compute perplexity on the prompt: `exp(mean(−log p(next token)))`. Lower means the model finds the text less surprising.

Defaults (edit in `main.cpp`): 50 tokens, temperature `0.9`, prompt `"what is the weather today?"`.

---

#### Getting started

#### Requirements

- A C++17 compiler (GCC or Clang) and CMake ≥ 3.15
- Linux or macOS (uses `mmap`)
- Python ≥ 3.10 with `numpy` and `huggingface_hub`
- ~500 MB of disk space for the float32 weights

#### 1. Install Python dependencies

With [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

or with pip:

```bash
pip install -r requirements.txt
```

#### 2. Convert the weights

```bash
python scripts/convert_weight.py
mkdir -p model
mv model.bin model_index.json model/
```

The script writes to the current directory; the C++ runtime expects the files in `model/`.

Optionally, verify the conversion:

```bash
python scripts/verify_conversion.py
```

#### 3. Provide the vocabulary

The runtime needs `model/vocab.bin`: GPT-2's 50257 tokens in id order, each stored as one length byte followed by the token's raw UTF-8 bytes. No script in this repository generates it yet — see [Known limitations](#known-limitations).

#### 4. Build and run

```bash
./setup.sh
```

This deletes any existing `build/`, configures with CMake, compiles, and runs the binary from the repository root. Manually:

```bash
cmake -S . -B build
cmake --build build -j
./build/gpt_inference_engine     # run from the repo root so model/ is found
```

The build uses `-O2 -march=native`, so the binary is optimized for the machine it was built on and may not run on older CPUs.

---

#### Int8 quantization (experimental)

`scripts/absmax.py` applies absmax quantization per tensor: `scale = 128 / max|w|`, `q = round(w · scale)`, stored as int8 — a 4× size reduction. `load_gpt2_quant.cpp` and `model_quant.cpp` mirror the float engine using `Tensor_Quant`.

This path is not wired into `main.cpp` and is not yet correct; see below.

---

#### Known limitations

- `vocab.bin` generation is missing. A script to export GPT-2's vocabulary in the expected format still needs to be added.
- Tokenizer is approximate.Greedy longest-match, not true BPE with merge ranks.
- Hardcoded model. Dimensions, layer count and tensor offsets are fixed to GPT-2 small.
- Int8 path is unfinished:
  - `absmax.py` uses `128 / max|w|`, so the largest weight rounds to 128 and overflows int8 (should be `127`).
  - Activation scales in `model_quant.cpp` are placeholder constants (`0.01`) and some buffers' scales are never initialized, so outputs are not meaningful yet.
- Tokens/sec is overstated. The count includes prompt tokens, not only generated ones.
- Single-threaded. No multithreading of the matrix-vector products yet.

