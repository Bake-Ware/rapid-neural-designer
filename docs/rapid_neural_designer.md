# Rapid Neural Designer (RND)
## Visual Programming Interface for Neural VM Research

**Version:** 1.0
**Date:** October 1, 2025
**Project Status:** Functional Web-Based Tool

---

## Overview

**Rapid Neural Designer (RND)** is a Scratch-like visual programming interface for building neural network experiments through drag-and-drop blocks. It generates executable Python code from visual block compositions, with full integration into the Neural VM framework for complete computational state capture.

### Key Innovation

Unlike traditional neural network builders that only create model architectures, **RND generates code that captures full computational state** - all intermediate tensors, attention patterns, Q/K/V projections, and semantic metadata - enabling deep introspection into network behavior.

---

## Core Features

### 🎨 Visual Block Programming
- **Scratch-like interface** using Google Blockly
- Drag-and-drop atomic components (Linear, Attention, LayerNorm, etc.)
- No code writing required for basic experimentation
- Visual organization of experiment structure

### ⚡ Real-Time Code Generation
- **Live Python code** generated as you build
- Monaco Editor with syntax highlighting
- Automatic imports and component definitions
- Instant code preview in "Code" tab

### ✓ Smart Validation
- **Automatic linting** for component connections
- Tensor dimension mismatch detection (basic heuristics)
- Workspace validation feedback
- Warning/error highlighting in code editor

### 💾 Save/Load System
- **Export workspace as XML** for later editing
- Import saved experiments
- Portable experiment definitions
- Version control friendly

### 📥 Export & Execution
- **Download generated Python** to run locally
- Copy code to clipboard
- Optional backend execution (Python Flask server)
- Real-time execution output display

---

## Architecture

### Technology Stack

```
Frontend:
├── Google Blockly (block-based visual programming)
├── Monaco Editor (VS Code's editor, syntax highlighting)
├── Pure JavaScript (no build tools)
└── HTML/CSS (dark theme UI)

Backend (Optional):
├── Python Flask (code execution server)
├── subprocess execution sandbox
└── stdout/stderr capture
```

### Key Design Decisions

1. **Client-Side Everything**: No server required for basic use - runs entirely in browser
2. **Zero Install**: Open `index.html` in any modern browser, no dependencies
3. **Monaco Editor Integration**: Professional IDE experience with linting, autocomplete, syntax highlighting
4. **Blockly Custom Blocks**: Purpose-built blocks for Neural VM atomic components
5. **PyTorch + NumPy Code Gen**: Generates code compatible with both frameworks

---

## Available Block Categories

### 🧪 Experiment Structure
**Purpose:** Organize experiment workflow

- **Neural VM Experiment**: Main container with Setup/Components/Execute sections
- **Create Component**: Define and name neural components
- **Forward Pass**: Execute computation through components
- **Print State**: Display captured computational state

**Example Usage:**
```
🧪 Neural VM Experiment "attention_test"
  ├─ Setup: Set random seed, device selection
  ├─ Components: Define Linear, Attention layers
  └─ Execute: Run forward passes, print states
```

### 🔢 Atomic Components
**Purpose:** Neural network building blocks

- **Linear Layer**: `SimpleLinearAtom(in_features, out_features, bias)`
- **Multi-Head Attention**: `SimpleAttentionAtom(embed_dim, num_heads)`
- **Layer Norm**: `LayerNormAtom(normalized_shape)`
- **Activation**: ReLU, GELU, Tanh, Sigmoid
- **Add (Residual)**: Tensor addition for residual connections
- **Dropout**: Regularization layer

**Code Generation Example:**
```python
# Block: Linear Layer (in: 512, out: 256, bias: true)
layer1 = SimpleLinearAtom(512, 256, bias=True)
```

### 📊 Data & Variables
**Purpose:** Input creation and variable management

- **Input Tensor**: Random tensor generation with configurable batch/seq/dim
- **Embedding**: Token embedding layer
- **Positional Encoding**: Sinusoidal position embeddings
- **Variable**: Reference variables from experiment

**Code Generation Example:**
```python
# Block: Input Tensor (batch: 2, seq: 10, dim: 512)
input_tensor = np.random.randint(0, 1000, (2, 10))

# Block: Embedding (vocab: 50257, dim: 512)
embedding_layer = EmbeddingAtom(50257, 512)
```

### 💾 File I/O
**Purpose:** Dataset and model checkpoint management

- **Load Dataset**: HuggingFace, CSV, NumPy, Text, JSON
- **Save Data**: Export numpy arrays
- **Load Pretrained**: HuggingFace model loading
- **Save/Load Checkpoint**: Model state persistence

**Code Generation Example:**
```python
# Block: Load Dataset (HuggingFace, "wikitext-2")
dataset = load_dataset('wikitext-2')

# Block: Save Checkpoint (path: "model.pth")
torch.save(model.state_dict(), 'model.pth')
```

### 🎓 Training
**Purpose:** Gradient descent and optimization

- **Optimizer**: Adam, AdamW, SGD, RMSprop
- **Loss Function**: CrossEntropy, MSE, L1, BCE
- **Compute Loss**: Loss calculation
- **Backward Pass**: Gradient computation
- **Optimizer Step**: Parameter update
- **Zero Gradients**: Gradient clearing

**Code Generation Example:**
```python
# Block: Optimizer (Adam, lr: 0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Block: Compute Loss
loss = loss_fn(predictions, targets)

# Block: Backward Pass
loss.backward()

# Block: Optimizer Step
optimizer.step()
```

### 🔁 Control Flow
**Purpose:** Loops and iteration

- **For Range**: Standard for loop
- **For Each**: Iterate over collections

**Code Generation Example:**
```python
# Block: For i in range 10
for i in range(10):
    # nested blocks here
    pass
```

### 🛠️ Utilities
**Purpose:** Debugging and configuration

- **Print Shape**: Debug tensor dimensions
- **Set Random Seed**: Reproducibility
- **Device**: CPU/CUDA/MPS selection
- **To Device**: Tensor device transfer

**Code Generation Example:**
```python
# Block: Set Random Seed (42)
np.random.seed(42)
torch.manual_seed(42)

# Block: Device (Auto)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 🎭 Multimodal
**Purpose:** Advanced multimodal model support (Qwen3-Omni, etc.)

- **Load Multimodal Model**: HuggingFace multimodal models
- **Load Processor**: Tokenizer/processor for multimodal inputs
- **Conversation**: Chat template creation
- **Message**: Text/multimodal messages
- **Process Multimodal**: Input preprocessing
- **Generate Multimodal Response**: Text/audio generation
- **Decode Response**: Output decoding
- **Save Audio**: Audio file export

**Code Generation Example:**
```python
# Block: Load Multimodal Model ("Qwen/Qwen3-Omni-30B-A3B-Instruct")
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2"
)

# Block: Generate Multimodal Response (max_tokens: 1024, speaker: "Ethan")
output = model.generate(**inputs, max_new_tokens=1024, speaker="Ethan")
```

---

## User Interface

### Layout

```
┌─────────────────────────────────────────────────────────┐
│  🧠 Neural VM Builder                                   │
│  [Designer] [Code]    💾 📁 🗑️ 📋 ⬇️                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Designer Tab:                                          │
│  ┌─────────────┬─────────────────────────────────────┐ │
│  │  Toolbox    │  Workspace (drag blocks here)       │ │
│  │             │                                     │ │
│  │ 🧪 Experiment│                                     │ │
│  │ 🔢 Components│                                     │ │
│  │ 📊 Data      │                                     │ │
│  │ 💾 File I/O  │                                     │ │
│  │ 🎓 Training  │                                     │ │
│  │ 🔁 Control   │                                     │ │
│  │ 🛠️ Utilities │                                     │ │
│  │ 🎭 Multimodal│                                     │ │
│  │ 🔧 Variables │                                     │ │
│  └─────────────┴─────────────────────────────────────┘ │
│                                                         │
│  Code Tab:                                              │
│  ┌───────────────────────────────────────────────────┐ │
│  │  Generated Python Code       [▶️ Run Code]        │ │
│  ├───────────────────────────────────────────────────┤ │
│  │  Monaco Editor (syntax highlighting)              │ │
│  │  import numpy as np                               │ │
│  │  import torch                                     │ │
│  │  ...                                              │ │
│  ├───────────────────────────────────────────────────┤ │
│  │  Validation Messages                              │ │
│  │  ✓ 2 atomic component(s) defined                 │ │
│  │  ⚠️ Dimension mismatch: expected 256, got 512    │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Controls

| Button | Action |
|--------|--------|
| **💾 Save** | Export workspace as XML |
| **📁 Load** | Import saved workspace XML |
| **🗑️ Clear** | Clear entire workspace (with confirmation) |
| **📋 Copy Code** | Copy generated Python to clipboard |
| **⬇️ Download** | Download Python file |
| **▶️ Run Code** | Execute code on backend server (if running) |

---

## Code Generation

### Generated Code Structure

```python
# Neural VM Experiment: my_experiment
# Generated by Neural VM Builder

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import time

# Atomic Components (placeholder implementations)
class SimpleLinearAtom(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    def forward(self, x): return self.linear(x)

class SimpleAttentionAtom(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    def forward(self, x): return self.attn(x, x, x)[0]

# ... (additional component definitions)

# Setup
np.random.seed(42)
torch.manual_seed(42)

# Component definitions
layer1 = SimpleLinearAtom(512, 256, bias=True)
attention_layer = SimpleAttentionAtom(512, 8)

# Execution
input_tensor = np.random.randn(2, 10, 512)
output, state = layer1.forward(input_tensor)
print(f"Captured {state.component_type} state: {state.get_full_state_size()} elements")

print("Experiment 'my_experiment' completed successfully!")
```

### Component Classes

RND generates **placeholder PyTorch implementations** for atomic components. These are simplified versions meant for quick prototyping. For production use, replace with full Neural VM atomic components from `neuralAtomLib.py`.

**Placeholder vs. Full Implementation:**

| Aspect | Placeholder (Generated) | Full NVM (neuralAtomLib.py) |
|--------|------------------------|----------------------------------|
| State Capture | None | Complete (20+ state types) |
| QKV Preservation | No | Yes (Q/K/V projections saved) |
| Semantic Intent | No | Yes (metadata tracking) |
| Computational Trajectory | No | Yes (operation sequence) |
| Attention Analytics | No | Yes (entropy, concentration, etc.) |

---

## Usage Workflow

### 1. Build Experiment Visually

```
Step 1: Drag "🧪 Neural VM Experiment" to workspace
Step 2: Add blocks to Setup section (Set Random Seed, Device)
Step 3: Add blocks to Components section (Create Linear, Create Attention)
Step 4: Add blocks to Execute section (Forward Pass, Print State)
```

### 2. Review Generated Code

```
Step 1: Click "Code" tab
Step 2: Review Monaco Editor output
Step 3: Check validation messages
Step 4: Fix any warnings/errors
```

### 3. Export or Execute

**Option A: Export for Local Execution**
```
Step 1: Click "⬇️ Download" button
Step 2: Save .py file
Step 3: Run locally: python nvm_experiment.py
```

**Option B: Execute in Browser (requires backend)**
```
Step 1: Start backend: python backend.py
Step 2: Click "▶️ Run Code" button
Step 3: View output in Execution Output panel
```

---

## Integration with Neural VM

### State Capture Integration

RND generates code compatible with Neural VM's state capture system. To enable full state capture:

**Replace placeholder implementations:**

```python
# Instead of generated placeholder:
class SimpleLinearAtom(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    def forward(self, x): return self.linear(x)

# Use full NVM implementation:
from simple_experiment import SimpleLinearAtom, ComputationalState

# Now forward passes return (output, state) tuples with full state capture
```

### Example: Full State Capture

```python
# Generated by RND with placeholder:
layer1 = SimpleLinearAtom(512, 256, bias=True)
output = layer1.forward(input_tensor)

# Modified for full NVM state capture:
from simple_experiment import SimpleLinearAtom
layer1 = SimpleLinearAtom(512, 256, bias=True)
output, state = layer1.forward(input_tensor)

print(f"State size: {state.get_full_state_size()} elements")
print(f"Semantic intent: {state.semantic_intent}")
print(f"Trajectory: {state.computational_trajectory}")
print(f"Intermediate states: {list(state.intermediate_states.keys())}")
```

---

## Validation System

### Workspace Validation

**Checks performed:**
- Workspace not empty
- Has Experiment block
- Has at least one atomic component
- Component connections valid

**Feedback Examples:**
```
✓ 2 atomic component(s) defined
✓ Experiment looks good! Ready to generate code
⚠️ Workspace is empty. Start by dragging an Experiment block
⚠️ No atomic components added yet
```

### Code Validation (Monaco Linter)

**Checks performed:**
- Imports at top of file
- Forward pass results assigned to variables
- Basic tensor dimension mismatches

**Example Markers:**
```
Line 45: ⚠️ Forward pass result is not assigned to a variable
Line 78: ❌ Dimension mismatch: expected input 256, got 512
```

---

## Backend Execution (Optional)

### Flask Backend Server

**File:** `backend.py` (not shown, but referenced in UI)

**Features:**
- Executes generated Python code in subprocess sandbox
- Captures stdout/stderr
- Returns execution results as JSON
- Health check endpoint

**API Endpoints:**

```
GET  /health     → {"status": "ok", "message": "Backend ready"}
POST /execute    → {"success": bool, "stdout": str, "stderr": str, "execution_time": float, "error": str}
```

**Security Considerations:**
- Subprocess execution (isolated from main process)
- Timeout enforcement (prevent infinite loops)
- **NOT production-ready** - for local experimentation only

---

## Use Cases

### 1. Rapid Prototyping

**Scenario:** Test attention layer configurations quickly

```
1. Drag Experiment block
2. Add Input Tensor (batch=1, seq=10, dim=512)
3. Add Multi-Head Attention (heads=8)
4. Add Multi-Head Attention (heads=4)
5. Compare state capture sizes
6. Download code, run locally
```

**Benefit:** No code writing, instant visualization

### 2. Educational Demonstrations

**Scenario:** Teach transformer architecture

```
1. Build transformer block step-by-step
2. Show intermediate tensor shapes
3. Explain attention mechanism visually
4. Export for students to run
```

**Benefit:** Visual learning, concrete code output

### 3. Experiment Documentation

**Scenario:** Save experimental architecture configurations

```
1. Build complex multi-layer network
2. Save as XML (version control)
3. Share with collaborators
4. Load and modify later
```

**Benefit:** Reproducible, shareable, versionable

### 4. Quick Testing

**Scenario:** Verify dimension compatibility

```
1. Build network with specific dimensions
2. Check validation warnings
3. Fix mismatches visually
4. Generate correct code
```

**Benefit:** Catch errors before writing code

---

## Limitations & Future Work

### Current Limitations

1. **Simplified State Capture**: Generated code uses placeholder implementations, not full NVM state capture
2. **Limited Validation**: Basic dimension checking, not comprehensive type/shape analysis
3. **No Visual State Display**: Can't visualize captured state in browser (yet)
4. **Backend Security**: Execution backend is local-only, not production-safe
5. **Fixed Block Set**: Can't add custom atomic components via UI (requires code modification)

### Planned Enhancements

**Phase 1 (Near-term):**
- Enhanced dimension mismatch detection
- Visual state visualization (attention heatmaps, tensor shapes)
- More atomic components (RNN, Mamba, CNN, GNN)
- Improved error messages

**Phase 2 (Medium-term):**
- Context bus operations (Phase 2 integration)
- Trail state visualization (cognitive backtracking UI)
- Live execution in browser (WebAssembly/PyScript)
- Custom block creation UI

**Phase 3 (Long-term):**
- Abstraction MLP integration (peak detection visualization)
- Cross-architecture translation UI
- Collaborative editing (multi-user workspaces)
- Cloud execution backend (safe sandboxing)

---

## Technical Implementation Details

### Blockly Configuration

**Theme:** Custom dark theme matching VS Code aesthetics

```javascript
theme: Blockly.Theme.defineTheme('dark', {
    'base': Blockly.Themes.Classic,
    'componentStyles': {
        'workspaceBackgroundColour': '#1e1e1e',
        'toolboxBackgroundColour': '#252526',
        'flyoutBackgroundColour': '#2d2d30',
        // ...
    }
})
```

**Workspace Configuration:**
- Grid: 20px spacing with snap-to-grid
- Zoom: 0.3x to 3x range
- Trashcan: Enabled for block deletion
- Toolbox: Category-based with collapsible sections

### Code Generator Pattern

Each block type has a `forBlock` generator function:

```javascript
pythonGenerator.forBlock['nvm_linear'] = function(block, generator) {
    const inFeatures = block.getFieldValue('IN_FEATURES');
    const outFeatures = block.getFieldValue('OUT_FEATURES');
    const useBias = block.getFieldValue('USE_BIAS') === 'TRUE';

    const code = `SimpleLinearAtom(${inFeatures}, ${outFeatures}, bias=${useBias})`;
    return [code, generator.ORDER_FUNCTION_CALL];
};
```

**Key Methods:**
- `block.getFieldValue(name)` - Get field values (numbers, dropdowns, text)
- `generator.valueToCode(block, inputName, order)` - Get connected block's code
- `generator.statementToCode(block, inputName)` - Get nested statement blocks' code
- `return [code, order]` for value blocks
- `return code` for statement blocks

### Monaco Editor Integration

**Setup:**
```javascript
require(['vs/editor/editor.main'], function() {
    monacoEditor = monaco.editor.create(document.getElementById('monaco-editor'), {
        value: '# Initial code...',
        language: 'python',
        theme: 'vs-dark',
        automaticLayout: true,
        minimap: { enabled: true },
        // ... IDE features
    });
});
```

**Features Enabled:**
- Syntax highlighting (Python language server)
- Autocomplete (word-based suggestions)
- Parameter hints
- Format on paste/type
- Minimap navigation
- Context menu (cut/copy/paste)

---

## File Structure

```
web_interface/
├── index.html              # Main application (2000 lines)
│   ├── HTML structure
│   ├── CSS styling (dark theme)
│   ├── Blockly block definitions
│   ├── Python code generators
│   ├── Monaco Editor setup
│   └── UI control logic
│
├── validate_xml.js         # XML workspace validation
└── (optional) backend.py   # Flask execution server
```

---

## Installation & Setup

### Zero Installation Mode (Recommended for Quick Start)

```bash
# Simply open the file in a browser:
cd C:\neural_vm_experiments\web_interface
start index.html

# Or use Python's built-in server:
python -m http.server 8000
# Then open: http://localhost:8000/index.html
```

**No dependencies, no build tools, no installation required.**

### Full Setup with Backend Execution

```bash
# 1. Install Python dependencies
pip install flask flask-cors

# 2. Start backend server
cd C:\neural_vm_experiments\web_interface
python backend.py

# 3. Open index.html in browser
start index.html

# Backend will be accessible at http://localhost:5000
```

---

## Example Workflows

### Example 1: Simple Attention Test

**Visual Blocks:**
```
🧪 Neural VM Experiment "attention_test"
  Setup:
    - Set Random Seed (42)
  Components:
    - Create layer1 = Multi-Head Attention (embed_dim: 512, heads: 8)
  Execute:
    - Input Tensor (batch: 1, seq: 10, dim: 512) → x
    - Forward Pass: output = layer1(x)
    - Print State (state)
```

**Generated Code:**
```python
import numpy as np
import torch
# ... imports ...

np.random.seed(42)
torch.manual_seed(42)

layer1 = SimpleAttentionAtom(512, 8)

x = np.random.randint(0, 1000, (1, 10))
output, state = layer1.forward(x)
print(f"Captured {state.component_type} state: {state.get_full_state_size()} elements")
print(f"State: {state}")
```

### Example 2: Transformer Block

**Visual Blocks:**
```
🧪 Neural VM Experiment "transformer_block"
  Components:
    - Create attn = Multi-Head Attention (512, heads: 8)
    - Create ln1 = Layer Norm (512)
    - Create ffn = Linear Layer (512 → 2048)
    - Create activation = Activation (GELU)
    - Create ffn_out = Linear Layer (2048 → 512)
    - Create ln2 = Layer Norm (512)
    - Create residual = Add (Residual)
  Execute:
    - Input Tensor (1, 10, 512) → x
    - Forward: attn_out = attn(x)
    - Forward: normed1 = ln1(residual(x, attn_out))
    - Forward: ffn_hidden = activation(ffn(normed1))
    - Forward: ffn_final = ffn_out(ffn_hidden)
    - Forward: output = ln2(residual(normed1, ffn_final))
```

**Benefit:** Visually construct complex architectures, immediately see Python implementation

---

## Relation to Neural VM Research

### RND's Role in the Ecosystem

**RND is the user interface layer** for the Neural VM research project:

```
┌──────────────────────────────────────┐
│  Rapid Neural Designer (RND)         │  ← User Interface
│  Visual block programming            │
└─────────────┬────────────────────────┘
              ↓ generates
┌──────────────────────────────────────┐
│  Python Code with Atomic Components  │  ← Generated Code
│  SimpleLinearAtom, SimpleAttentionAtom│
└─────────────┬────────────────────────┘
              ↓ uses (when replaced)
┌──────────────────────────────────────┐
│  Neural VM Core (neuralAtomLib.py)│  ← State Capture
│  Complete state preservation          │
└─────────────┬────────────────────────┘
              ↓ stores in
┌──────────────────────────────────────┐
│  Hyperbolic Manifold (Future)        │  ← Context Bus
│  Universal state storage              │
└──────────────────────────────────────┘
```

### Integration Points

1. **Code Generation**: RND generates PyTorch code compatible with NVM atomic components
2. **State Capture**: Generated code structure supports (output, state) tuple returns
3. **Experimentation**: RND enables rapid testing of architectures that will use NVM state capture
4. **Visualization** (Future): RND will visualize captured state from NVM experiments

---

## Future Vision

### Towards a Unified Neural IDE

**Long-term Goal:** RND becomes a comprehensive visual IDE for neural research with:

1. **Live State Visualization**: Attention heatmaps, tensor flow diagrams, activation distributions
2. **Interactive Debugging**: Step through forward passes, inspect intermediate states
3. **Cross-Architecture Translation UI**: Drag transformer blocks, convert to CNN visually
4. **Abstraction Playground**: Visualize peak detection, adjust abstraction levels interactively
5. **Cognitive Trail Explorer**: Navigate reasoning checkpoints, explore alternative paths
6. **Collaborative Workspaces**: Real-time multi-user experimentation

**Analogy:** Jupyter Notebooks meets Scratch meets TensorBoard, purpose-built for Neural VM research.

---

## Document Metadata

**Author**: Neural VM Research Team
**Version**: 1.0
**Last Updated**: October 1, 2025
**Status**: Production-Ready Web Tool
**Next Review**: After Phase 2 Context Bus integration

---

**RND: Making neural network experimentation as easy as playing with LEGO blocks.** 🧱🧠
