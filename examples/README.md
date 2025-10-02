# Noun Classifier Example

A working demonstration of Rapid Neural Designer's full computational state capture using a simple "is it a noun?" classifier.

## What This Demonstrates

This example shows RND's key differentiator: **capturing complete computational state**, not just model outputs. Unlike typical neural networks that only preserve final predictions, RND captures:

- Complete Q/K/V attention projections
- Attention patterns and weights
- All intermediate transformations
- Weight matrices and biases
- Semantic intent and computational trajectories

## Quick Start

### 0. Try in RND Visual Designer (Optional)

Open `../web_interface/index.html` in your browser and load `noun_classifier_workspace.xml` to see the model structure visually and generate the code yourself!

### 1. Generate Dataset (1000+ words)

```bash
python generate_dataset.py
```

Creates `noun_dataset.json` with ~1000 common English words labeled as noun/non-noun.

### 2. Verify State Capture Works

```bash
python verify_state_capture.py
```

Runs a quick test to verify all computational states are being captured properly. Should output:

```
SUCCESS: Full computational state capture verified!
```

### 3. Train the Classifier

```bash
python train_noun_classifier.py
```

Trains a simple attention-based classifier. Note: The model doesn't learn well (uses simplified gradients), but that's not the point - **the point is demonstrating state capture during real ML operations**.

### 4. Test Individual Words

```bash
python test_noun.py dog
python test_noun.py quickly
python test_noun.py computer
```

Analyzes a word and shows:
- Prediction and confidence
- All 3 captured computational states (attention → projection → classification)
- Attention patterns over characters
- Total state size captured (~35KB per inference)
- Exports full state to JSON

## Files

- `noun_classifier_workspace.xml` - **Load this in RND UI to see the model!**
- `generate_dataset.py` - Creates the training dataset
- `train_noun_classifier.py` - Training script using atomic components
- `test_noun.py` - CLI tool to test words and inspect state
- `verify_state_capture.py` - Verification that state capture works
- `noun_dataset.json` - Generated training data (1065 words)

## Architecture

```
Input: "computer"
    ↓
[Character Embedding] → (1, 20, 28) one-hot vectors
    ↓
[Attention Layer] → Captures Q/K/V states, attention patterns
    ↓
[Pooling] → Mean over sequence
    ↓
[Linear Projection] → Captures weight matrices, transformations
    ↓
[Classifier] → Binary output (noun/not noun)
```

Each component uses RND's atomic building blocks that capture full internal state.

## State Captured Per Inference

- **Attention Component**: ~7,500 values
  - Q, K, V projections (4 heads × 20 seq × 7 dims each)
  - Attention weights (4 heads × 20×20)
  - Attention scores, entropy, concentration metrics

- **Linear Layers**: ~1,600 values
  - Weight matrices and biases
  - Input/output statistics
  - Transformation magnitudes

**Total: ~9,100 float values (~36KB) of complete computational state**

## Why This Matters

Traditional neural networks are black boxes - you get a prediction but can't see the internal reasoning. RND makes the full computational process transparent:

1. **Interpretability**: See exactly what the attention focuses on
2. **Debugging**: Inspect intermediate states when things go wrong
3. **Research**: Analyze how information flows through the network
4. **Cross-Architecture**: Captured state can potentially be translated to other architectures

This is the foundation for the larger Neural VM vision - preserving semantic information across different computational paradigms.

## Limitations

- Model doesn't actually learn well (simplified gradient descent)
- Character-level encoding is naive
- No proper train/test split
- Just a proof of concept for state capture

The goal isn't to build a great noun classifier - it's to demonstrate that RND captures **everything** happening inside the neural network during computation.
