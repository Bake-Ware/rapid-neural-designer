# Neural VM Builder - Web Interface

A Scratch-like visual programming interface for building Neural VM experiments using drag-and-drop atomic components.

## Features

- **Visual Block Programming**: Drag and drop neural network components like Scratch
- **Real-time Code Generation**: See Python code generated as you build
- **Validation**: Basic linting for component connections and tensor dimensions
- **Save/Load**: Export and import your experiments as XML
- **Export Code**: Download generated Python code to run locally

## Getting Started

1. Open `index.html` in any modern web browser (Chrome, Firefox, Edge, Safari)
2. No installation or server required - runs entirely in the browser

## Available Blocks

### 🧪 Experiment Blocks
- **Neural VM Experiment**: Main container for your experiment
- **Create Component**: Define and name a component
- **Forward Pass**: Execute forward pass through a component
- **Print State**: Display captured computational state

### 🔢 Atomic Components
- **Linear Layer**: Linear transformation with configurable dimensions and bias
- **Multi-Head Attention**: Attention mechanism with Q/K/V state capture

### 📊 Data Blocks
- **Input Tensor**: Create random input tensors with configurable shape
- **Variable**: Reference variables created in your experiment

## Example: Building a Simple Transformer Layer

1. Drag an **"🧪 Neural VM Experiment"** block to the workspace
2. Inside "Setup", add an **Input Tensor** block (e.g., batch=1, seq=10, dim=512)
3. Inside "Components":
   - Add **"Create Component"** blocks for attention and feedforward layers
   - Configure a **Multi-Head Attention** (embed_dim=512, heads=8)
   - Configure a **Linear Layer** (in=512, out=512)
4. Inside "Execute":
   - Add **Forward Pass** blocks to run data through your components
5. Click **"⬇️ Download"** to get the Python code

## Generated Code

The interface generates Python code compatible with the Neural VM atomic components from `simple_experiment.py`. The generated code includes:
- Proper imports (numpy, dataclasses, etc.)
- Component initialization
- Forward passes with state capture
- State inspection and logging

## Controls

- **💾 Save**: Export workspace as XML file
- **📁 Load**: Import previously saved workspace
- **🗑️ Clear**: Clear the entire workspace
- **📋 Copy Code**: Copy generated Python to clipboard
- **⬇️ Download**: Download generated Python file

## Validation Messages

The interface provides real-time validation:
- ⚠️ **Warnings**: Missing components or empty workspace
- ✓ **Success**: Valid experiment structure detected

## Future Enhancements

- Tensor dimension mismatch detection
- More atomic components (RNN, Mamba, CNN, etc.)
- Phase 2 blocks (peak detector, abstraction MLP)
- Visual state visualization
- Context bus operations
- Control flow blocks (loops, conditionals)

## Technical Details

- Built with Google Blockly (Scratch's web equivalent)
- Pure JavaScript - no build tools required
- Generates Python code for numpy-based atomic components
- Saves/loads workspace as XML