# Neural VM Software Design Document
## Atomic Neural Network Framework with Context-Aware Abstraction

**Version:** 2.0  
**Date:** September 29, 2025  
**Status:** Phase 1 Complete, Phase 2 Design

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Current Implementation Status](#current-implementation-status)
3. [Architecture Design](#architecture-design)
4. [Abstraction System Design](#abstraction-system-design)
5. [Implementation Roadmap](#implementation-roadmap)
6. [File Structure](#file-structure)

---

## 1. Project Overview

### 1.1 Vision
Build a Neural Virtual Machine (NVM) that enables cross-architecture neural computation through:
- Atomic neural components with complete state capture
- Universal context bus for lossless state preservation
- Context-aware abstraction for semantic compression
- Multi-level representation for cross-domain matching

### 1.2 Core Innovation
Instead of lossy tensor handoffs, preserve full computational state (Q,K,V, attention patterns, trajectories, abstractions) in a hyperbolic space-based context bus.

### 1.3 Key Problems Solved
- **Cross-architecture translation**: Transformer → Diffusion, RNN → CNN
- **Semantic preservation**: Maintain meaning across computational paradigms
- **Temporal integration**: Handle RNN/Mamba/Liquid sequential models
- **Cognitive primitives**: Implement actual abstraction, not simulated abstraction

---

## 2. Current Implementation Status

### 2.1 Phase 1: COMPLETE ✅

**Location:** `C:\neural_vm_experiments\`

**Delivered:**
- Atomic component state capture (Linear, Attention)
- Empirical analysis framework
- Neural context bus specifications
- Hyperbolic space manifold design

**Key Files:**
```
C:\neural_vm_experiments\
├── neuralAtomLib.py          # Working state capture experiment
├── atomic_components.py          # PyTorch-based components (blocked by torch issues)
├── bus_requirements.json         # Generated specifications
├── phase1_results.md            # Experimental results
└── README.md                    # Project overview
```

**Validated Findings:**
- 20 unique state types captured
- QKV attention states fully preserved
- Hyperbolic space requirements determined:
  - Attention Manifold: 1024D, curvature -1.0
  - Transformation Manifold: 512D, curvature 0.0
  - Semantic Manifold: 256D, curvature -0.5

---

## 3. Architecture Design

### 3.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Neural VM Layer                          │
├─────────────────────────────────────────────────────────────┤
│  Atomic Components  →  State Capture  →  Context Bus        │
│       ↓                    ↓                  ↓              │
│  [Linear, Attention,  [Complete State]  [Hyperbolic        │
│   RNN, Mamba, etc]    [Trajectories]     Storage]          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Abstraction System (NEW)                       │
├─────────────────────────────────────────────────────────────┤
│  Peak Detection  →  Context-Aware MLP  →  Multi-Level      │
│                                           Embeddings        │
│  [Statistical]      [Learned Mapping]    [Semantic Space]  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│           Universal Semantic Representation                 │
├─────────────────────────────────────────────────────────────┤
│  Cross-Architecture • Cross-Linguistic • Cross-Domain       │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Unified State Object Design

**Design Decision:** Single unified state object (not separate manifolds)

**Rationale:**
- Enable heap strategy for state passing
- Support constructor pattern: `Component(previous_state)`
- Allow atomic state transitions (succeed/fail as unit)
- Enable state checkpointing and backtracking

```python
class UnifiedNeuralState:
    """Single object containing all computational state"""
    
    def __init__(self, total_dimension: int = 1792):
        # Unified embedding space with manifold regions
        self.embedding_space = np.zeros((1, total_dimension))
        
        # Manifold slicing (logical separation, physical unity)
        self.attention_region = slice(0, 1024)
        self.transform_region = slice(1024, 1536)
        self.semantic_region = slice(1536, 1792)
        
        # Trail state for cognitive backtracking
        self.reasoning_trail = []
        self.decision_checkpoints = []
        
        # Abstraction state
        self.abstraction_levels = {}
        self.peak_indices = []
```

### 3.3 Trail State & Cognitive Backtracking

**Motivation:** Current models simulate "ah-ha moments" without actual backtracking capability. Implement real cognitive revert functionality.

**Design:**
```python
@dataclass
class ReasoningCheckpoint:
    """Snapshot of cognitive state at decision point"""
    timestamp: float
    state_snapshot: np.ndarray
    decision_metadata: Dict[str, Any]
    alternative_paths: List[str]
    assumptions_made: List[str]
    confidence_score: float
    
class TrailState:
    """Complete reasoning history for backtracking"""
    checkpoints: List[ReasoningCheckpoint]
    current_path: List[str]
    explored_alternatives: Set[str]
    
    def backtrack_to(self, checkpoint_id: str) -> UnifiedNeuralState:
        """Revert to previous cognitive state"""
        pass
        
    def explore_alternative(self, branch_point: str) -> UnifiedNeuralState:
        """Explore different reasoning path from checkpoint"""
        pass
```

---

## 4. Abstraction System Design

### 4.1 Two-Stage Abstraction Pipeline

**Stage 1: Statistical Peak Detection**  
**Stage 2: Context-Aware MLP Abstraction**

### 4.2 Peak Detection Module

**File:** `C:\neural_vm_experiments\abstraction\peak_detector.py` (NEW)

**Purpose:** Deterministic extraction of semantic "edges" from neural activations

**Algorithm:**
```python
class PeakDetector:
    """Statistical abstraction through activation peak detection"""
    
    def __init__(self, threshold_percentile: float = 90):
        self.threshold = threshold_percentile
        
    def extract_peaks(self, activations: np.ndarray, 
                     attention_weights: Optional[np.ndarray] = None) -> PeakSet:
        """
        Extract high-attention tokens/features
        
        Returns:
            PeakSet containing:
            - peak_indices: Locations of peaks
            - peak_values: Activation strengths
            - peak_embeddings: Embedding vectors
            - context_embeddings: Surrounding context
        """
        
        # Method 1: Attention-based (for transformers)
        if attention_weights is not None:
            peaks = self._attention_peaks(attention_weights)
        
        # Method 2: Activation-based (for CNNs, MLPs)
        else:
            peaks = self._activation_peaks(activations)
        
        return PeakSet(
            indices=peaks,
            embeddings=activations[peaks],
            context=self._extract_context(activations, peaks)
        )
    
    def _attention_peaks(self, attention_weights: np.ndarray) -> np.ndarray:
        """Extract tokens with highest attention values"""
        # Aggregate across heads and layers
        aggregated = attention_weights.mean(axis=(0, 1))  # Average over heads
        threshold_value = np.percentile(aggregated, self.threshold)
        return np.where(aggregated >= threshold_value)[0]
    
    def _activation_peaks(self, activations: np.ndarray) -> np.ndarray:
        """Extract features with highest activation magnitudes"""
        activation_norms = np.linalg.norm(activations, axis=-1)
        threshold_value = np.percentile(activation_norms, self.threshold)
        return np.where(activation_norms >= threshold_value)[0]
```

### 4.3 Context-Aware Abstraction MLP

**File:** `C:\neural_vm_experiments\abstraction\abstraction_mlp.py` (NEW)

**Purpose:** Learn contextual abstraction patterns (mimics human abstraction development)

**Architecture:**
```python
class AbstractionMLP:
    """
    Learns context-aware semantic abstraction
    
    Input: [word_embedding, context_embedding, abstraction_level]
    Output: target_embedding_coordinates
    
    Training: Learn from (specific, general) pairs with context
    """
    
    def __init__(self, embedding_dim: int = 512, 
                 context_dim: int = 256,
                 hidden_dim: int = 1024):
        
        # Network architecture
        self.input_layer = LinearLayer(embedding_dim + context_dim + 1, hidden_dim)
        self.hidden1 = LinearLayer(hidden_dim, hidden_dim)
        self.hidden2 = LinearLayer(hidden_dim, hidden_dim)
        self.output_layer = LinearLayer(hidden_dim, embedding_dim)
        
        # Activation functions
        self.activation = nn.GELU()  # Smooth, differentiable
        
    def forward(self, word_embedding: np.ndarray,
                context_embedding: np.ndarray,
                abstraction_level: float) -> np.ndarray:
        """
        Compute abstraction target based on context
        
        Args:
            word_embedding: Specific concept embedding (e.g., "dog")
            context_embedding: Semantic context (e.g., biology vs. household)
            abstraction_level: 0.0 (specific) to 1.0 (maximally general)
            
        Returns:
            target_embedding: Where to move in embedding space
        """
        
        # Concatenate inputs
        x = np.concatenate([
            word_embedding,
            context_embedding,
            [abstraction_level]
        ])
        
        # Forward pass
        x = self.activation(self.input_layer(x))
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        target = self.output_layer(x)
        
        return target
    
    def abstract(self, word_embedding: np.ndarray,
                context_embedding: np.ndarray,
                level: float = 0.5) -> np.ndarray:
        """
        Apply learned abstraction
        
        Returns:
            Abstracted embedding in semantic space
        """
        target = self.forward(word_embedding, context_embedding, level)
        
        # Interpolate toward target based on abstraction level
        abstracted = word_embedding + level * (target - word_embedding)
        
        return abstracted
```

### 4.4 Training Strategy

**File:** `C:\neural_vm_experiments\abstraction\train_abstraction.py` (NEW)

**Training Data Sources:**
1. **WordNet Hierarchies**: (dog, mammal), (chase, pursue)
2. **Wikipedia Categories**: (Python, Programming Language, Technology)
3. **Domain Ontologies**: Scientific, culinary, military taxonomies
4. **Human Examples**: Crowdsourced abstraction pairs with context

**Curriculum Learning:**
```python
class AbstractionTrainer:
    """Train abstraction MLP with curriculum learning"""
    
    def __init__(self, mlp: AbstractionMLP):
        self.mlp = mlp
        self.optimizer = Adam(learning_rate=0.001)
        
    def train(self, training_phases: List[TrainingPhase]):
        """
        Phase 1: Basic abstractions (dog→animal)
        Phase 2: Contextual abstractions (dog-biology→mammal)
        Phase 3: Complex abstractions (democracy→governance-system)
        """
        
        for phase in training_phases:
            print(f"Training Phase: {phase.name}")
            
            for epoch in range(phase.num_epochs):
                for batch in phase.data_loader:
                    loss = self._train_step(batch)
                    
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def _train_step(self, batch: TrainingBatch):
        """
        Single training step
        
        Loss: MSE between predicted target and actual abstraction embedding
        """
        
        predictions = []
        targets = []
        
        for example in batch:
            # Predict abstraction target
            pred_target = self.mlp(
                example.specific_embedding,
                example.context_embedding,
                example.abstraction_level
            )
            predictions.append(pred_target)
            
            # Actual general concept embedding
            targets.append(example.general_embedding)
        
        # Compute loss and update
        loss = mse_loss(predictions, targets)
        self.optimizer.step(loss)
        
        return loss
```

### 4.5 Integration with State Capture

**File:** `C:\neural_vm_experiments\abstraction\integrated_capture.py` (NEW)

```python
class AbstractionAwareStateCapture:
    """Extended state capture with multi-level abstraction"""
    
    def __init__(self):
        self.peak_detector = PeakDetector(threshold_percentile=90)
        self.abstraction_mlp = AbstractionMLP()
        
    def capture_with_abstraction(self, 
                                component_state: ComputationalState,
                                abstraction_levels: List[float] = [0.0, 0.3, 0.5, 0.7, 1.0]
                                ) -> MultiLevelState:
        """
        Capture state at multiple abstraction levels
        
        Returns:
            MultiLevelState containing:
            - level_0: Original peaks (no abstraction)
            - level_1-4: Progressively abstracted representations
            - embedding_trajectories: Path through embedding space
        """
        
        # Extract peaks from activations/attention
        peaks = self.peak_detector.extract_peaks(
            component_state.output_tensor,
            component_state.attention_patterns
        )
        
        # Extract context from non-peak regions
        context = self._compute_context(
            component_state.output_tensor,
            exclude_indices=peaks.indices
        )
        
        # Generate multi-level abstractions
        abstracted_states = {}
        
        for level in abstraction_levels:
            abstracted_embeddings = []
            
            for peak_embedding in peaks.embeddings:
                # Apply learned abstraction
                abstracted = self.abstraction_mlp.abstract(
                    peak_embedding,
                    context,
                    level=level
                )
                abstracted_embeddings.append(abstracted)
            
            abstracted_states[f"level_{level}"] = np.array(abstracted_embeddings)
        
        return MultiLevelState(
            original_state=component_state,
            peak_indices=peaks.indices,
            abstraction_levels=abstracted_states,
            context_embedding=context
        )
```

---

## 5. Implementation Roadmap

### 5.1 Phase 2: Abstraction System (Current)

**Timeline:** 2-3 weeks

**Deliverables:**
1. ✅ Design document (this document)
2. ⬜ Peak detection implementation
3. ⬜ Abstraction MLP architecture
4. ⬜ Training data pipeline
5. ⬜ Initial training runs
6. ⬜ Validation experiments

**Milestones:**
- Week 1: Implement peak detector and test on Phase 1 data
- Week 2: Build and train abstraction MLP
- Week 3: Integration with state capture system

### 5.2 Phase 3: Temporal Integration

**Timeline:** 2-3 weeks

**Deliverables:**
1. ⬜ Temporal manifold implementation
2. ⬜ RNN/Mamba state capture
3. ⬜ Trail state and backtracking
4. ⬜ Multi-head confidence convergence

### 5.3 Phase 4: Cross-Architecture Translation

**Timeline:** 3-4 weeks

**Deliverables:**
1. ⬜ Neural bytecode specification
2. ⬜ Compiler/interpreter for different VMs
3. ⬜ Semantic fidelity metrics
4. ⬜ Cross-architecture experiments

---

## 6. File Structure

### 6.1 Current Structure
```
C:\neural_vm_experiments\
│
├── README.md                    # Project overview
├── phase1_results.md           # Experimental results
├── bus_requirements.json       # Context bus specifications
│
├── neuralAtomLib.py        # ✅ Working state capture
├── atomic_components.py        # ⚠️ PyTorch version (blocked)
├── bus_analysis.py            # Analysis framework
└── quick_test.py              # Test harness
```

### 6.2 Proposed Structure (Phase 2+)

```
C:\neural_vm_experiments\
│
├── README.md
├── phase1_results.md
├── DESIGN_DOC.md              # This document
├── bus_requirements.json
│
├── core\                      # Core VM components
│   ├── __init__.py
│   ├── atomic_components.py   # Base component classes
│   ├── state_capture.py       # State instrumentation
│   └── unified_state.py       # UnifiedNeuralState class
│
├── abstraction\               # NEW: Abstraction system
│   ├── __init__.py
│   ├── peak_detector.py       # Statistical peak extraction
│   ├── abstraction_mlp.py     # Context-aware MLP
│   ├── train_abstraction.py   # Training pipeline
│   ├── integrated_capture.py  # Combined system
│   └── models\                # Saved model checkpoints
│       └── abstraction_mlp_v1.npz
│
├── context_bus\               # Neural context bus
│   ├── __init__.py
│   ├── hyperbolic_space.py    # Manifold implementations
│   ├── storage.py             # State storage system
│   └── addressing.py          # Cross-manifold addressing
│
├── temporal\                  # Temporal models support
│   ├── __init__.py
│   ├── rnn_capture.py         # RNN state capture
│   ├── mamba_capture.py       # Mamba/SSM capture
│   ├── trail_state.py         # Cognitive backtracking
│   └── temporal_manifold.py   # Temporal storage
│
├── experiments\               # Experimental code
│   ├── phase1_state_capture.py    # ✅ Original experiment
│   ├── phase2_abstraction_test.py # Peak detection validation
│   ├── phase2_mlp_training.py     # MLP training experiments
│   └── results\
│       ├── phase1\
│       └── phase2\
│
├── training_data\             # Training datasets
│   ├── wordnet_hierarchies.json
│   ├── wikipedia_categories.json
│   └── domain_ontologies\
│       ├── scientific.json
│       ├── culinary.json
│       └── military.json
│
└── tests\                     # Unit tests
    ├── test_peak_detector.py
    ├── test_abstraction_mlp.py
    └── test_integrated_system.py
```

---

## 7. Technical Specifications

### 7.1 Dependencies
```
numpy>=1.24.0
scipy>=1.10.0
# torch>=2.0.0  # Currently blocked, fallback to numpy
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

### 7.2 Performance Requirements

**Peak Detection:**
- Throughput: >1000 tokens/second
- Latency: <10ms per sequence
- Memory: <100MB for typical inputs

**Abstraction MLP:**
- Inference: <5ms per embedding
- Training: ~1 hour for basic curriculum
- Model size: <50MB

**Context Bus:**
- Storage capacity: 10,000+ states in memory
- Retrieval latency: <1ms
- Compression ratio: 60-75% (via abstraction)

### 7.3 Evaluation Metrics

**Abstraction Quality:**
- Semantic similarity preservation (cosine similarity)
- Human evaluation of abstraction appropriateness
- Cross-domain matching accuracy

**System Performance:**
- State capture completeness (all critical info preserved)
- Abstraction compression ratio
- Cross-architecture translation fidelity

---

## 8. Research Contributions

### 8.1 Novel Contributions

1. **Deterministic Statistical Abstraction**: Peak detection as semantic edge extraction
2. **Learned Context-Aware Abstraction**: MLP mimics human abstraction development
3. **Multi-Level Semantic Representation**: Continuous abstraction hierarchy
4. **Unified State Object**: Enables heap strategy and cognitive backtracking
5. **Trail State Architecture**: Real cognitive revert capability vs. simulated reflection

### 8.2 Paper Potential

**Paper 1**: "Universal Neural State Preservation: An Empirical Analysis"
- Phase 1 results
- State capture methodology
- Hyperbolic space requirements

**Paper 2**: "Context-Aware Semantic Abstraction for Neural Architectures"
- Peak detection algorithm
- Abstraction MLP design
- Multi-level representation system

**Paper 3**: "Neural Virtual Machine: Cross-Architecture Semantic Computation"
- Complete system integration
- Cross-architecture translation
- Semantic fidelity evaluation

---

## 9. Open Questions & Future Work

### 9.1 Research Questions

1. **Peak Detection Validation**: Do real transformer attention weights concentrate on semantic cores?
2. **Abstraction Universality**: Do learned abstraction patterns transfer across domains?
3. **Temporal Integration**: How to handle liquid neural networks' continuous dynamics?
4. **Semantic Fidelity**: Can we quantify meaning preservation across translations?

### 9.2 Future Extensions

- **Continuous Cognitive Loop**: Always-on background processing (daydreaming, memory consolidation)
- **Graph Memory Integration**: Connect to your previous graph-based memory work
- **Multi-Modal Abstraction**: Extend to images, audio, video
- **Distributed State**: Scale context bus across multiple machines

---

## 10. Getting Started

### 10.1 Run Phase 1 Experiment
```bash
cd C:\neural_vm_experiments
python neuralAtomLib.py
```

### 10.2 Begin Phase 2 Development
```bash
# Create new directories
mkdir abstraction
mkdir abstraction\models

# Start with peak detector
# Implement: abstraction\peak_detector.py
```

### 10.3 Training Data Preparation
```bash
# Download WordNet
# Extract Wikipedia categories
# Build domain ontologies
```

---

**Document Status:** Draft v1.0  
**Next Review:** After peak detector implementation  
**Owner:** Neural VM Research Team
