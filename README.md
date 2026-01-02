### Gpt inference engine Architecture

A high performance C/C++ inference engine that runs on CPU.
An inference engine is the runtime system responsible for executing a trained model to produce output. Inference engine runs the model efficiently.

In simple terms is the os, copiler and scheduler for neural network execution.

This engine solves latency, throughput, memory efficieny and hardware utilization (CPU) in this case (I got no GPU).

#### Functions:
1. Model Excecution: Runs the foward pass, applies attention, projections e.t.c

2. Memory Management: Manages weights, manages KV cache

3. Hardware Optimization: Uses CUDA kernels

#### High Level Architecture

```
Python (model prep & quantization)
        ↓
Serialized weights (binary)
        ↓
C++ Inference Runtime
        ↓
Tokenizer → Transformer → Sampling

```

