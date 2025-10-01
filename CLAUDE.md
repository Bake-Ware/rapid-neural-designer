# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Neural Virtual Machine (NVM) research project that explores cross-architecture neural computation through atomic components and a universal context bus. The goal is to preserve full computational state (Q,K,V states, attention patterns, semantic intent, etc.) across different neural architectures, enabling true cross-paradigm neural computation with semantic preservation.

## Key Concepts

- **Atomic Components**: Instrumented neural building blocks (Linear, Attention, RNN, etc.) that capture ALL internal state during computation
- **Context Bus**: Hyperbolic space-based storage system for lossless state preservation
- **Abstraction System**: Two-stage pipeline (statistical peak detection + learned context-aware MLP) for semantic compression
- **Unified State Object**: Single object containing all computational state, enabling heap strategy for state passing and cognitive backtracking

## Architecture

The system uses a multi-layer architecture:
1. **Atomic Components** → capture complete computational state during forward passes
2. **State Capture System** → records input/output tensors, intermediate states, attention patterns, semantic trajectories
3. **Hyperbolic Space Storage** → three manifolds for different state types:
   - Attention Manifold (1024D, curvature -1.0): Q/K/V states, attention weights
   - Transformation Manifold (512D, curvature 0.0): weight matrices, transformation vectors
   - Semantic Manifold (256D, curvature -0.5): semantic intents, computational trajectories
4. **Abstraction System** (Phase 2) → peak detection + context-aware MLP for multi-level semantic representation

## Commands

### Running Experiments

```bash
# Phase 1 state capture experiment (working, numpy-based)
python simple_experiment.py

# Quick test harness
python quick_test.py

# Bus analysis framework
python bus_analysis.py
```

### Development

```bash
# The project uses Python 3.12+
python --version

# No package manager setup yet - dependencies are:
# numpy>=1.24.0, scipy>=1.10.0, scikit-learn>=1.3.0, matplotlib>=3.7.0
# Note: PyTorch support exists but is currently blocked - use numpy implementations
```

## File Structure

**Core Implementation:**
- `simple_experiment.py` - Working numpy-based state capture experiment (Phase 1 complete)
- `atomic_components.py` - PyTorch-based instrumented components (currently blocked by torch issues, use numpy version)
- `bus_analysis.py` - Experimental framework to analyze storage requirements

**Specifications & Results:**
- `bus_requirements.json` - Generated specifications from Phase 1 experiments (20 state types, 3 manifold designs)
- `phase1_results.md` - Experimental results and findings
- `DESIGN_DOC.md` - Complete technical design (architecture, abstraction system, roadmap)
- `README.md` - Project overview and vision

## Development Guidelines

### State Capture

All atomic components must implement complete state capture:
- Input/output tensors with shapes
- All intermediate computational states (Q/K/V projections, attention scores, etc.)
- Parameter states (weights, biases)
- Semantic metadata (intent, computational trajectory)
- Attention patterns when applicable

The `ComputationalState` dataclass is the standard container for captured state.

### Numpy vs PyTorch

- **Current Implementation**: Use numpy for all new development (`simple_experiment.py` is the reference)
- **Future**: PyTorch implementations exist in `atomic_components.py` but are blocked by environment issues
- When implementing components: provide both forward pass computation and state capture in single method

### Abstraction System (Phase 2 - Upcoming)

The planned abstraction system has two stages:
1. **Peak Detection** (`abstraction/peak_detector.py`): Statistical extraction of high-attention tokens/features
2. **Context-Aware MLP** (`abstraction/abstraction_mlp.py`): Learned semantic abstraction using word_embedding + context_embedding + abstraction_level → target_embedding

Training will use WordNet hierarchies, Wikipedia categories, and domain ontologies.

## Phase Status

- ✅ **Phase 1**: Atomic component state capture complete
  - 20 unique state types identified
  - Hyperbolic space requirements determined
  - Bus specifications generated

- 🔄 **Phase 2**: Abstraction system design complete, implementation pending
  - Peak detector design ready
  - Abstraction MLP architecture specified
  - Training pipeline planned

- ⬜ **Phase 3**: Temporal integration (RNN/Mamba state capture, trail state, cognitive backtracking)
- ⬜ **Phase 4**: Cross-architecture translation (neural bytecode, compiler/interpreter)

## Important Implementation Notes

1. **Unified State Object**: Components should accept and return `UnifiedNeuralState` objects (not yet implemented but specified in DESIGN_DOC.md)
2. **Heap Strategy**: State passing uses constructor pattern: `Component(previous_state)`
3. **Trail State**: Future support for cognitive backtracking with reasoning checkpoints (not simulated reflection)
4. **Semantic Intent**: Every component must declare its semantic intent (e.g., "linear_transformation", "selective_attention_mechanism")
5. **Computational Trajectories**: Track the sequence of operations (e.g., "input_projection" → "qkv_projection" → "attention_score_computation")

## Research Context

This project positions itself as solving fundamental infrastructure problems in AI:
- Enabling hybrid architectures (Transformer + Diffusion, CNN + Graph Neural Networks)
- Preserving semantic fidelity across computational paradigms
- Implementing true abstraction (not simulated)
- Supporting genuine cognitive backtracking (not token-based reflection)

The work builds toward cross-modal reasoning where transformer attention states can guide diffusion model generation, CNN feature hierarchies inform graph neural networks, etc.