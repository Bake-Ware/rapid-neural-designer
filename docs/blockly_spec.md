# RND Blockly Code Generation Specification

This document defines the exact behavior of each block and the code it should generate in each supported language.

## Core Principles

1. **XML is Universal**: The same XML must generate semantically equivalent code in all languages
2. **No Placeholders**: Generated code uses real `neuralAtomLib` in all languages
3. **Type Safety**: Blocks validate connections (Tensor→Tensor, Component→Component)
4. **Runnable Output**: Generated code must run without manual editing

---

## Block Catalog

### 1. `nvm_experiment` - Main Container

**Purpose**: Top-level experiment wrapper

**Inputs**:
- `EXP_NAME` (field): Name of experiment
- `SETUP` (statement): Variable initialization (optional)
- `COMPONENTS` (statement): Component creation
- `EXECUTE` (statement): Forward passes and execution

**Output**: Complete program

**Python Generation**:
```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import time

from neuralAtomLib import (
    SimpleLinearAtom,
    SimpleAttentionAtom,
    ComputationalState
)

# Setup
{SETUP_CODE}

# Component definitions
{COMPONENTS_CODE}

# Execution
{EXECUTE_CODE}

print("Experiment '{EXP_NAME}' completed successfully!")
```

**C# Generation**:
```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using RapidNeuralDesigner;

namespace NeuralVMExperiment
{
    class Program
    {
        static void Main(string[] args)
        {
            // Setup
{SETUP_CODE}

            // Component definitions
{COMPONENTS_CODE}

            // Execution
{EXECUTE_CODE}

            Console.WriteLine("Experiment '{EXP_NAME}' completed successfully!");
        }
    }
}
```

---

### 2. `nvm_create_component` - Component Instantiation

**Purpose**: Create and name a neural component

**Inputs**:
- `VAR_NAME` (field): Variable name for component
- `COMPONENT` (value): Component block (nvm_linear, nvm_attention, etc.)

**Output**: Statement

**Python Generation**:
```python
{VAR_NAME} = {COMPONENT_CODE}
```

**C# Generation**:
```csharp
var {VAR_NAME} = {COMPONENT_CODE};
```

**Example XML**:
```xml
<block type="nvm_create_component">
  <field name="VAR_NAME">my_layer</field>
  <value name="COMPONENT">
    <block type="nvm_linear">
      <field name="IN_FEATURES">64</field>
      <field name="OUT_FEATURES">32</field>
      <field name="USE_BIAS">TRUE</field>
    </block>
  </value>
</block>
```

**Expected Output (Python)**:
```python
my_layer = SimpleLinearAtom(64, 32, bias=True)
```

**Expected Output (C#)**:
```csharp
var my_layer = new SimpleLinearAtom(64, 32, bias: true);
```

---

### 3. `nvm_linear` - Linear Layer

**Purpose**: Linear transformation component

**Inputs**:
- `IN_FEATURES` (field): Input dimension
- `OUT_FEATURES` (field): Output dimension
- `USE_BIAS` (field): TRUE/FALSE

**Output**: Value (Component type)

**Python Generation**:
```python
SimpleLinearAtom({IN_FEATURES}, {OUT_FEATURES}, bias={USE_BIAS_BOOL})
```

**C# Generation**:
```csharp
new SimpleLinearAtom({IN_FEATURES}, {OUT_FEATURES}, bias: {USE_BIAS_BOOL})
```

**Boolean Conversion**:
- Python: `TRUE` → `True`, `FALSE` → `False`
- C#: `TRUE` → `true`, `FALSE` → `false`

---

### 4. `nvm_attention` - Multi-Head Attention

**Purpose**: Attention mechanism component

**Inputs**:
- `EMBED_DIM` (field): Embedding dimension (must be divisible by num_heads)
- `NUM_HEADS` (field): Number of attention heads

**Output**: Value (Component type)

**Python Generation**:
```python
SimpleAttentionAtom(d_model={EMBED_DIM}, num_heads={NUM_HEADS})
```

**C# Generation**:
```csharp
new SimpleAttentionAtom(dModel: {EMBED_DIM}, numHeads: {NUM_HEADS})
```

---

### 5. `nvm_input` - Input Tensor Creation

**Purpose**: Create random input tensor for testing

**Inputs**:
- `BATCH_SIZE` (field): Batch dimension
- `SEQ_LEN` (field): Sequence length
- `EMBED_DIM` (field): Embedding dimension

**Output**: Value (Tensor type)

**Python Generation**:
```python
np.random.randn({BATCH_SIZE}, {SEQ_LEN}, {EMBED_DIM}).astype(np.float32)
```

**C# Generation**:
```csharp
torch.randn(new long[] {{{BATCH_SIZE}, {SEQ_LEN}, {EMBED_DIM}}})
```

**Usage**: This is a **value block**, must be used inline or assigned to variable

**Valid**:
```xml
<block type="nvm_create_component">
  <field name="VAR_NAME">input_data</field>
  <value name="COMPONENT">
    <block type="nvm_input">...</block>
  </value>
</block>
```

**OR**:
```xml
<block type="nvm_forward">
  <value name="INPUT">
    <block type="nvm_input">...</block>
  </value>
</block>
```

---

### 6. `nvm_forward` - Forward Pass

**Purpose**: Execute forward pass through component

**Inputs**:
- `OUTPUT_VAR` (field): Variable name for output
- `COMPONENT_VAR` (field): Variable name of component to call
- `INPUT` (value): Input tensor (from nvm_input or nvm_variable)

**Output**: Statement

**Python Generation**:
```python
{OUTPUT_VAR}, state = {COMPONENT_VAR}.forward({INPUT_CODE})
print(f"Captured {{state.component_type}} state: {{state.get_full_state_size()}} elements")
```

**C# Generation**:
```csharp
var ({OUTPUT_VAR}, state) = {COMPONENT_VAR}.forward({INPUT_CODE});
Console.WriteLine($"Captured {{state.ComponentType}} state: {{state.GetFullStateSize()}} elements");
```

---

### 7. `nvm_variable` - Variable Reference

**Purpose**: Reference an existing variable

**Inputs**:
- `VAR` (field): Variable name

**Output**: Value (any type)

**Python Generation**:
```python
{VAR}
```

**C# Generation**:
```csharp
{VAR}
```

---

### 8. `nvm_print_state` - Debug Output

**Purpose**: Print variable value

**Inputs**:
- `STATE_VAR` (field): Variable to print

**Output**: Statement

**Python Generation**:
```python
print(f"State: {{{STATE_VAR}}}")
```

**C# Generation**:
```csharp
Console.WriteLine($"State: {{{STATE_VAR}}}");
```

---

## Complete Example

### XML
```xml
<xml xmlns="https://developers.google.com/blockly/xml">
  <block type="nvm_experiment" x="20" y="20">
    <field name="EXP_NAME">test_classifier</field>
    <statement name="COMPONENTS">
      <block type="nvm_create_component">
        <field name="VAR_NAME">layer1</field>
        <value name="COMPONENT">
          <block type="nvm_linear">
            <field name="IN_FEATURES">64</field>
            <field name="OUT_FEATURES">32</field>
            <field name="USE_BIAS">TRUE</field>
          </block>
        </value>
      </block>
    </statement>
    <statement name="EXECUTE">
      <block type="nvm_forward">
        <field name="OUTPUT_VAR">output</field>
        <field name="COMPONENT_VAR">layer1</field>
        <value name="INPUT">
          <block type="nvm_input">
            <field name="BATCH_SIZE">2</field>
            <field name="SEQ_LEN">10</field>
            <field name="EMBED_DIM">64</field>
          </block>
        </value>
        <next>
          <block type="nvm_print_state">
            <field name="STATE_VAR">output</field>
          </block>
        </next>
      </block>
    </statement>
  </block>
</xml>
```

### Expected Python Output
```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import time

from neuralAtomLib import (
    SimpleLinearAtom,
    SimpleAttentionAtom,
    ComputationalState
)

# Component definitions
layer1 = SimpleLinearAtom(64, 32, bias=True)

# Execution
output, state = layer1.forward(np.random.randn(2, 10, 64).astype(np.float32))
print(f"Captured {state.component_type} state: {state.get_full_state_size()} elements")
print(f"State: {output}")

print("Experiment 'test_classifier' completed successfully!")
```

### Expected C# Output
```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using RapidNeuralDesigner;

namespace NeuralVMExperiment
{
    class Program
    {
        static void Main(string[] args)
        {
            // Component definitions
            var layer1 = new SimpleLinearAtom(64, 32, bias: true);

            // Execution
            var (output, state) = layer1.forward(torch.randn(new long[] {2, 10, 64}));
            Console.WriteLine($"Captured {state.ComponentType} state: {state.GetFullStateSize()} elements");
            Console.WriteLine($"State: {output}");

            Console.WriteLine("Experiment 'test_classifier' completed successfully!");
        }
    }
}
```

---

## Known Issues to Fix

1. **Python duplicate imports**: Old placeholder imports still present
2. **C# nameDB not initialized**: Missing variable name resolution
3. **SETUP block handling**: Should it even exist? Input tensors should be inline or in COMPONENTS
4. **Variable initialization**: Python generates `var = None` declarations - unnecessary

---

## Implementation Checklist

- [ ] Remove duplicate imports from Python generator
- [ ] Initialize C# generator nameDB properly
- [ ] Remove/deprecate SETUP block (use COMPONENTS for everything)
- [ ] Remove variable pre-declarations in Python
- [ ] Test round-trip: XML → Python → Run → Works
- [ ] Test round-trip: XML → C# → Compile → Works
