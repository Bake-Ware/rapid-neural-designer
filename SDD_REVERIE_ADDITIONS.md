# SDD: REVERIE Primitives for RND

**Author:** Bake
**Status:** Draft v1
**Date:** 2026-05-24
**Audience:** Claude Code (implementing agent)

---

## 1. Objective

Add ~27 new definitions to RND's component/atomic catalogs so REVERIE architectures can be expressed visually in the editor. This adds **Tier A** (atomic primitives, ~10) and **Tier B** (molecular components, ~17) as identified by the REVERIE architecture audit.

**Non-goal:** Tier C (runtime/orchestration — state buffers, graph-DB bridge, training loop, dream triggers) is **out of scope**. Those live in a separate orchestrator codebase.

## 2. Current State

### 2.1 Existing Atomic Categories

- `math/` — matmul, add, multiply, etc.
- `trig/` — sin, cos, tanh, sigmoid, relu, gelu, softmax
- `reduction/` — sum, mean, max, min, norm, variance
- `shape/` — reshape, transpose, concat, split, slice, repeat, expand_dims, squeeze, broadcast
- `comparison/` — greater_than, less_than, equal, where, masked_fill
- `init/` — random_normal, random_uniform, zeros, ones, constant, parameter, arange, triangular_mask
- `data/` — input_tensor, embedding_lookup, output, print_shape

### 2.2 Existing Molecular Components

- `rnn_cell`, `rope_attention`, `rmsnorm`, `rope`, `transformer_block`, `qwen_block`, `swiglu_ffn`, `dropout`, `mamba_block`, `layernorm`, `gqa_attention`, `llada_block`, `lstm_layer`, `orthogonal_transform`, `gat_layer`

### 2.3 Discovery Mechanism

- `web_interface/atomics/index.json` — maps category IDs to file lists
- `web_interface/atomics/{category}.json` — array of atomic defs (id, name, inputs, outputs, properties, code_template)
- `web_interface/components/index.json` — maps component IDs to file names
- `web_interface/components/{id}.json` — component defs (name, description, inputs, outputs, properties, graph decomposition)
- `rnd/component_catalog.py` — reads both catalogs, caches in memory

## 3. Tier A — Atomic Primitives (New `atomic/util` Category)

Each has: `id`, `name`, `description`, `inputs`, `outputs`, `properties`, `code_template` (Python codegen string).

### 3.1 Category: `util`

| id | name | description | inputs | outputs | properties |
|---|---|---|---|---|---|
| `identity` | Identity | Pass-through. x → x. Graph hygiene + debug. | `x: tensor` | `out: tensor` | none |
| `stop_gradient` | StopGradient | During backward, prevents gradient flow through this node. Zero grad output. | `x: tensor` | `out: tensor` | none |
| `tensor_split` | TensorSplit | Fans out one tensor to N consumers. No-op op; just for readable graph routing. | `x: tensor` | `out_0: tensor`, `out_1: tensor` | `num_outputs: int (default 2)` |
| `conditional_gate` | ConditionalGate | where(cond, a, b). Element-wise select. | `cond: tensor`, `a: tensor`, `b: tensor` | `out: tensor` | none |
| `weighted_mix` | WeightedMix | α·x + (1-α)·y. Interpolation. | `x: tensor`, `y: tensor` | `out: tensor` | `alpha: float (default 0.5)` |

### 3.2 Additions to `math` category

| id | name | description | inputs | outputs | properties |
|---|---|---|---|---|---|
| `pc_error` | PCError | Computes prediction_error = observation − prediction. Routes error for use as both signal and learning trigger. | `observation: tensor`, `prediction: tensor` | `error: tensor` | none |
| `forward_noising` | ForwardNoise | x + σ·ε where ε ~ N(0, I). Adds noise at specified level. | `x: tensor` | `noisy_x: tensor` | `sigma: float (default 1.0)` |
| `cfg_combiner` | CFGCombiner | w·cond_pred + (1-w)·uncond_pred. Classifier-free guidance. | `cond: tensor`, `uncond: tensor` | `out: tensor` | `guidance_scale: float (default 3.0)` |
| `free_energy` | FreeEnergy | Aggregates prediction error across layers into a single scalar via sum/mean. | `errors: tensor[]` | `energy: tensor` (scalar) | `reduction: enum (sum|mean, default sum)` |

### 3.3 Additions to `control` category (new)

| id | name | description | inputs | outputs | properties |
|---|---|---|---|---|---|
| `surprise_gate` | SurpriseGate | Binary trigger when |error| > threshold. Used for auto-triggering consolidation/witness decisions. | `error: tensor`, `threshold: float` | `threshold: float (default 0.5)` |

## 4. Tier B — Molecular Components

These compositions build on existing + Tier A atomics. Each component definition includes a `graph` field describing its internal wiring (compatible with RND's existing component JSON schema).

### 4.1 Diffusion & Sequence

| id | name | description | composition |
|---|---|---|---|
| `time_embedding` | TimeEmbedding | Sinusoidal position encoding for diffusion step k. Standard sin/cos at varying frequencies + learned linear projection. | `sin` + `cos` + `linear` |
| `noise_schedule` | NoiseSchedule | Parameterized β/α table (linear/cosine/sigmoid). Outputs α_t and σ_t for step t. | `arange` + `lookup` (table) |
| `prediction_head` | PredictionHead | Norm + linear projection. Configurable parameterization (ε / x0 / v). | `layernorm` + `linear` (with param variant) |
| `cfg_dropout` | CFGDropout | Training-time conditional masking. Drops conditioning token with probability p to enable CFG at inference. | `dropout` + `conditional_gate` |

### 4.2 Predictive Coding

| id | name | description | composition |
|---|---|---|---|
| `pc_generative_head` | PCGenerativeHead | Per-layer top-down predictor. Each PC layer needs a generative branch alongside feedforward: norm + linear + activation. | `layernorm` + `linear` + `relu` |
| `pc_state_update` | PCStateUpdate | Local PC update: x ← x − α·(e_top − e_bottom). The whole point of PC is this is not backprop. | `subtract` + `weighted_mix` (with learned α) |

### 4.3 Witness & Extraction

| id | name | description | composition |
|---|---|---|---|
| `witness_head` | WitnessHead | Calibrated confidence scalar over (k, latent). Norm + linear + sigmoid + consistency regularization hook (training-time). | `layernorm` + `linear` + `sigmoid` |
| `halt_decision` | HaltDecision | PonderNet-style halt probability. Gumbel-softmax sample for differentiability. | `witness_head` + `gumbel_softmax` (pseudo) |
| `trajectory_recorder` | TrajectoryRecorder | Captures (k, x_k, conf_k) tuples across denoising steps. Ring-buffer in tensor form. | `concat` + `slice` (rolling buffer) |
| `extract_operator` | ExtractOperator | Given trajectory + halt decision, returns extracted latent at the halt step via index select. | `gather` + `squeeze` |

### 4.4 Memory & GAD

| id | name | description | composition |
|---|---|---|---|
| `graph_attention_diffusion` | GAD | Iterated GAT with diffusion-style updates. Wraps existing `gat_layer` in a diffusion loop: x_{t+1} = GAT(x_t, A) with skip connection. | `gat_layer` + `add` + `weighted_mix` (loop unrolled to depth param) |
| `graph_node_retrieval` | GraphRetrieval | Top-k semantically similar nodes from store given query. Query/key similarity + top-k select. Differentiable via learned key/query projection. | `linear` + `matmul` + `softmax` + `topk` (pseudo) |
| `tag_embedding` | TagEmbedding | Semantic tag attached to each weight pocket. Standard embedding lookup. | `embedding_lookup` (existing atomic) |

### 4.5 Action & Curriculum

| id | name | description | composition |
|---|---|---|---|
| `dual_track_combiner` | DualTrackCombiner | Combines converged extracted state + witness-filtered partial-denoising candidates. The half-baked-as-feature mechanism. | `witness_head` + `conditional_gate` + `weighted_mix` |
| `droga_actor` | DROGAActor | Curriculum-conditioned action head. Selects from a distribution of candidate states rather than emitting a single converged token. | `linear` + `softmax` + `gather` |
| `outcome_predictor` | OutcomePredictor | At action commit, predicts expected outcome distribution. Required for PC error injection when result returns. | `linear` + `layernorm` |

### 4.6 Advanced (P3 — Post-MVP)

| id | name | description | composition |
|---|---|---|---|
| `belief_update` | BeliefUpdate | Bayesian latent-belief update given observation. Could be replaced by PC inference loop in most cases. | `multiply` + `normalize` (pseudo Bayesian) |
| `expected_free_energy` | ExpectedFreeEnergy | Friston-style EFE: pragmatic + epistemic value across candidate actions. | `free_energy` + `weighted_mix` |

## 5. Implementation Plan for Claude Code

### 5a. Add Tier A primitives

1. Create/update `web_interface/atomics/util.json` with 5 new atomic defs (identity, stop_gradient, tensor_split, conditional_gate, weighted_mix)
2. Create/update `web_interface/atomics/control.json` with 1 new atomic (surprise_gate)
3. Append to `web_interface/atomics/math.json`: pc_error, forward_noising, cfg_combiner, free_energy
4. Update `web_interface/atomics/index.json` — add `util` and `control` categories

Each atomic JSON format:
```json
{
  "id": "identity",
  "name": "Identity",
  "description": "Pass-through tensor. x → x.",
  "inputs": [{"name": "x", "type": "tensor"}],
  "outputs": [{"name": "out", "type": "tensor"}],
  "properties": [],
  "code_template": "{out} = {x}"
}
```

### 5b. Add Tier B components

1. Create component JSON files under `web_interface/components/` for 16 new components (list above, minus belief_update and expected_free_energy which are P3)
2. Update `web_interface/components/index.json`

Component JSON format:
```json
{
  "name": "time_embedding",
  "description": "Sinusoidal time embedding for diffusion steps",
  "inputs": [{"name": "step", "type": "int"}],
  "outputs": [{"name": "embedding", "type": "tensor", "shape": [d_model]}],
  "properties": [
    {"name": "d_model", "type": "int", "default": 512}
  ],
  "graph": {
    "nodes": [...],
    "edges": [...]
  }
}
```

### 5c. Legacy `atomic_components.py` stubs (optional)

If the RND code generator produces Python, update `atomic_components.py` or create a sibling module with stub implementations for codegen targets. Use numpy-compatible implementations.

### 5d. Validation

After adding, verify:
- `component_catalog.py` loads all new primitives without error
- Atomic categories appear in `list_atomic_categories()` and contain correct items
- `list_atomics()` returns all new atomics
- `get_atomic()` returns individual defs correctly
- `get_node_definition()` resolves both atomic and molecular lookups

## 6. Verification

Run this test sequence:
```python
from rnd.component_catalog import ComponentCatalog
from pathlib import Path

cat = ComponentCatalog(Path("web_interface"))

# Check atomics
cats = cat.list_atomic_categories()
assert any(c["id"] == "util" for c in cats), "util category missing"
assert any(c["id"] == "control" for c in cats), "control category missing"

atomics = cat.list_atomics()
for aid in ["identity", "stop_gradient", "tensor_split", "conditional_gate",
            "weighted_mix", "surprise_gate", "pc_error", "forward_noising",
            "cfg_combiner", "free_energy"]:
    assert any(a["id"] == aid for a in atomics), f"atomic {aid} missing"

# Check components
comps = cat.list_components()
for cid in ["time_embedding", "noise_schedule", "prediction_head",
            "cfg_dropout", "pc_generative_head", "pc_state_update",
            "witness_head", "halt_decision", "trajectory_recorder",
            "extract_operator", "graph_attention_diffusion",
            "graph_node_retrieval", "dual_track_combiner",
            "droga_actor", "outcome_predictor"]:
    assert any(c["id"] == cid for c in comps), f"component {cid} missing"

print("✅ All 27 REVERIE primitives verified in RND catalog")
```