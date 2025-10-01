# Running Qwen3-Omni with Neural VM Builder

## Overview

Qwen3-Omni-30B is a multimodal foundation model that supports:
- **Input**: Text, Images, Audio, Video
- **Output**: Text, Speech (10 languages, multiple voices)
- **Architecture**: MoE-based "Thinker-Talker" design with 35.3B parameters
- **Special Features**: Real-time audio/video interaction, 119 text languages, end-to-end multilingual

## Requirements

### Hardware
- **Minimum GPU Memory**: 78.85 GB for 15s video processing
- **Recommended**: A100/H100 or multiple GPUs with `device_map="auto"`
- **CPU Alternative**: Very slow, not recommended for production

### Software Dependencies

```bash
pip install transformers>=4.37.0
pip install torch>=2.0.0
pip install flash-attn>=2.0.0  # For flash_attention_2
pip install qwen-omni-utils
pip install soundfile
pip install Pillow
pip install numpy
```

## Building with Neural VM Builder

### Step 1: Load the Example

1. Open `index.html` in your browser
2. Click **📁 Load**
3. Select `qwen3_omni_example.xml`
4. The complete Qwen3-Omni inference pipeline will load

### Step 2: Understanding the Workflow

The example demonstrates:

**Setup:**
- Set random seed for reproducibility

**Components:**
- **Load Multimodal Model**: Downloads Qwen3-Omni from HuggingFace
  - Uses `auto` dtype (automatically selects best precision)
  - Uses `flash_attention_2` for efficiency
  - Device map auto-distributes across GPUs

- **Load Processor**: Loads the multimodal processor
  - Handles tokenization, image processing, audio encoding

**Execution:**
1. **Create Conversation**: Build multimodal conversation
   - Add user message with image + text
   - Supports multiple modalities per message

2. **Process Inputs**: Prepare data for model
   - Applies chat template
   - Processes images, audio, video
   - Creates model inputs

3. **Generate Response**: Run inference
   - Returns both text and audio
   - Can specify speaker voice (Ethan, Emma, Grace, Olivia, etc.)
   - Supports streaming for real-time interaction

4. **Decode Outputs**:
   - Extract text response
   - Extract audio response

5. **Save Results**:
   - Print text to console
   - Save audio to WAV file

### Step 3: Customize Your Experiment

#### Change Input Modalities

**Add Audio Input:**
```
Multimodal Input → type: Audio URL → content: https://example.com/audio.wav
```

**Add Video Input:**
```
Multimodal Input → type: Video Path → content: /path/to/video.mp4
```

**Mix Multiple Inputs:**
```
User Message:
  - Image Input
  - Audio Input
  - Text Input: "What do you see and hear?"
```

#### Adjust Generation Parameters

In the **Generate Multimodal Response** block:
- `max_tokens`: Control response length (default: 1024)
- `speaker`: Choose voice
  - English: Ethan (male), Emma (female), Grace (female), Olivia (female)
  - Chinese: Junjun (male), Luna (female), Xiaobei (male), Xiaoni (female)
  - Japanese: Takumi (male), Taeko (female)

#### Add Multi-Turn Conversation

Add multiple message blocks:
```
Message (system): "You are a helpful assistant."
Message (user): "Hello!"
Message (assistant): "Hi! How can I help?"
Message (user): [Image] "What's in this image?"
```

## Generated Code Structure

The builder generates production-ready Python code:

```python
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import soundfile as sf
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

# Neural VM Experiment: qwen3_omni_inference

# Setup
np.random.seed(42)
torch.manual_seed(42)

# Component definitions
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2"
)

processor = Qwen3OmniMoeProcessor.from_pretrained(
    "Qwen/Qwen3-Omni-30B-A3B-Instruct"
)

# Execution
conversation = []
conversation.append({
    "role": "user",
    "content": [
        {"type": "image", "image": "https://example.com/image.jpg"},
        {"type": "text", "text": "Describe this image in detail."}
    ]
})

# Process multimodal inputs
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)

inputs = processor(
    text=text,
    audio=audios,
    images=images,
    videos=videos,
    return_tensors="pt",
    padding=True,
    use_audio_in_video=True
)
inputs = inputs.to(model.device).to(model.dtype)

# Generate response
output = model.generate(
    **inputs,
    max_new_tokens=1024,
    speaker="Ethan",
    thinker_return_dict_in_generate=True,
    use_audio_in_video=True
)

# Decode outputs
response_text = processor.batch_decode(output[0], skip_special_tokens=True)[0]
response_audio = output[1]

# Save results
print(f"State: {response_text}")
sf.write("qwen_response.wav", response_audio.cpu().numpy(), 16000)
```

## Advanced Use Cases

### 1. Video Analysis
```
Multimodal Input → Video Path → /path/to/video.mp4
Multimodal Input → Text → "Summarize what happens in this video"
```

### 2. Audio Transcription + Analysis
```
Multimodal Input → Audio Path → interview.wav
Multimodal Input → Text → "Transcribe this audio and identify the speakers"
```

### 3. Real-Time Streaming (requires additional setup)
```python
# Use streaming mode for low-latency interaction
output = model.generate(
    **inputs,
    streamer=TextIteratorStreamer(processor),
    max_new_tokens=1024
)
```

### 4. Multi-Language Interaction
```
Message (user):
  - Text: "请用中文描述这张图片" (Chinese)
  - Image: image.jpg

# Model will respond in Chinese with Luna voice
```

## Memory Optimization

For systems with limited GPU memory:

1. **Use 4-bit quantization**:
```python
from transformers import BitsAndBytesConfig

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    device_map="auto"
)
```

2. **Reduce context length**:
- Use shorter videos/audio clips
- Resize images to smaller dimensions
- Limit conversation history

3. **Use gradient checkpointing** (for fine-tuning):
```python
model.gradient_checkpointing_enable()
```

## Neural VM State Capture

The Qwen3-Omni model captures:
- **Thinker States**: Internal reasoning process
- **Talker States**: Speech generation states
- **Vision Encoder States**: Image/video embeddings
- **Audio Encoder States**: Audio embeddings
- **Cross-Modal Attention**: How different modalities interact
- **MoE Router Decisions**: Which experts are activated

This enables:
- Analyzing how the model reasons multimodally
- Understanding attention patterns across modalities
- Debugging generation quality
- Cross-architecture translation (e.g., Qwen → GPT-4V)

## Troubleshooting

**Error: "CUDA out of memory"**
- Reduce max_tokens
- Use smaller batch size
- Enable 4-bit quantization
- Use shorter media inputs

**Error: "flash_attention_2 not found"**
```bash
pip install flash-attn --no-build-isolation
```

**Slow inference**
- Ensure flash_attention_2 is installed
- Use GPU (model is too large for CPU)
- Check device_map is using all available GPUs

**Audio quality issues**
- Verify sample rate matches model (16000 Hz)
- Check audio format is supported (WAV, FLAC, MP3)
- Ensure audio files aren't corrupted

## Next Steps

1. **Fine-tune** on your domain-specific data
2. **Integrate** into applications (chatbots, analysis tools)
3. **Experiment** with different speaker voices
4. **Combine** with other NVM atomic components for hybrid architectures
5. **Analyze** captured states for research insights

## Resources

- **Model Card**: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
- **Paper**: [Link when published]
- **Qwen GitHub**: https://github.com/QwenLM/Qwen
- **Neural VM Docs**: See CLAUDE.md in this repo