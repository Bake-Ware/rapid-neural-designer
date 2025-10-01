# Building a Simple GPT with Neural VM Builder

## Architecture Overview

A minimal GPT (Generative Pre-trained Transformer) consists of:

1. **Token Embedding** - Convert token IDs to dense vectors
2. **Positional Encoding** - Add position information
3. **Transformer Block** (repeated N times):
   - Multi-Head Self-Attention
   - Residual Connection + Layer Norm
   - Feed-Forward Network (FFN)
   - Residual Connection + Layer Norm
4. **Final Layer Norm**
5. **Output Projection** - Project to vocabulary size

## How to Build in Neural VM Builder

### Step 1: Setup Embeddings

1. Drag **"🧪 Neural VM Experiment"** block
2. In the **Components** section:
   - Add **"Create Component"** block, name it `embedding`
     - Connect **"Embedding"** block (vocab: 50257, dim: 512)
   - Add **"Create Component"** block, name it `pos_encoding`
     - Connect **"Positional Encoding"** block (max_len: 512, dim: 512)

### Step 2: Build One Transformer Block

For each transformer block, you need:

**Attention Sub-layer:**
1. Create `attention` component:
   - **Multi-Head Attention** (embed_dim: 512, heads: 8)
2. Create `attn_norm` component:
   - **Layer Norm** (dims: 512)
3. Create `add1` component:
   - **Add (Residual)** - for skip connection

**Feed-Forward Sub-layer:**
4. Create `ffn_linear1` component:
   - **Linear Layer** (in: 512, out: 2048, bias: true)
5. Create `ffn_activation` component:
   - **Activation** (GELU)
6. Create `ffn_linear2` component:
   - **Linear Layer** (in: 2048, out: 512, bias: true)
7. Create `ffn_norm` component:
   - **Layer Norm** (dims: 512)
8. Create `add2` component:
   - **Add (Residual)** - for skip connection

**Optional Regularization:**
9. Create `dropout` component:
   - **Dropout** (rate: 0.1)

### Step 3: Final Output Layer

1. Create `final_norm` component:
   - **Layer Norm** (dims: 512)
2. Create `output_projection` component:
   - **Linear Layer** (in: 512, out: 50257, bias: false)

### Step 4: Execution Flow

In the **Execute** section, add **Forward Pass** blocks in this order:

```
1. Forward Pass: embedded = embedding(input_tokens)
2. Forward Pass: x = pos_encoding(embedded)
3. Forward Pass: attn_out = attention(x)
4. Forward Pass: x = add1(x, attn_out)  # Residual connection
5. Forward Pass: x = attn_norm(x)
6. Forward Pass: ffn_out = ffn_linear1(x)
7. Forward Pass: ffn_out = ffn_activation(ffn_out)
8. Forward Pass: ffn_out = ffn_linear2(ffn_out)
9. Forward Pass: x = add2(x, ffn_out)  # Residual connection
10. Forward Pass: x = ffn_norm(x)
11. Forward Pass: x = final_norm(x)
12. Forward Pass: logits = output_projection(x)
```

## Minimal GPT Configuration

For a verifiable baseline GPT:

- **Vocabulary Size**: 50,257 (GPT-2 tokenizer)
- **Embedding Dimension**: 512
- **Number of Heads**: 8 (512 / 8 = 64 dims per head)
- **Number of Layers**: 6-12 (start with 6 for testing)
- **FFN Hidden Size**: 2048 (typically 4x embedding dim)
- **Context Length**: 512 tokens
- **Dropout**: 0.1

## State Capture

The Neural VM will capture:
- **Q, K, V projections** from each attention layer
- **Attention weights** (which tokens attend to which)
- **Intermediate activations** in FFN
- **Layer norms statistics** (mean, variance)
- **Residual stream** at each skip connection

## Generated Python Code Structure

The builder will generate code like:

```python
import numpy as np
# ... imports ...

# Neural VM Experiment: simple_gpt

# Component definitions
embedding = EmbeddingAtom(50257, 512)
pos_encoding = PositionalEncodingAtom(512, 512)
attention = SimpleAttentionAtom(512, 8)
attn_norm = LayerNormAtom(512)
add1 = AddAtom()
ffn_linear1 = SimpleLinearAtom(512, 2048, bias=True)
ffn_activation = ActivationAtom('gelu')
ffn_linear2 = SimpleLinearAtom(2048, 512, bias=True)
ffn_norm = LayerNormAtom(512)
add2 = AddAtom()
final_norm = LayerNormAtom(512)
output_projection = SimpleLinearAtom(512, 50257, bias=False)

# Execution
embedded, state = embedding.forward(input_tokens)
print(f"Captured {state.component_type} state: {state.get_full_state_size()} elements")
x, state = pos_encoding.forward(embedded)
print(f"Captured {state.component_type} state: {state.get_full_state_size()} elements")
# ... etc
```

## Verification Steps

1. **Shape Check**: Verify tensor dimensions match at each layer
2. **Attention Patterns**: Inspect captured Q, K, V states
3. **State Preservation**: Confirm all intermediate states are captured
4. **Context Bus**: Verify states can be stored in hyperbolic manifolds
5. **Cross-Architecture**: Test if captured states can inform other model types

## Next Steps

Once this baseline works:
1. Stack multiple transformer blocks
2. Add causal masking for autoregressive generation
3. Implement training loop
4. Test cross-architecture translation (Transformer → RNN/Mamba)
5. Verify semantic preservation across translations