"""
CLI tool to test noun classification and inspect captured computational state.
This demonstrates RND's full state capture capability.

Usage:
    python test_noun.py word

Example:
    python test_noun.py dog
    python test_noun.py quickly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from train_noun_classifier import NounClassifier
import json


def print_state_summary(state, indent="  "):
    """Print a human-readable summary of computational state."""
    print(f"{indent}Component: {state.component_type} ({state.component_id})")
    print(f"{indent}Intent: {state.semantic_intent}")
    print(f"{indent}Input shape: {state.input_shape}")
    print(f"{indent}Output shape: {state.output_shape}")

    # Print key intermediate states
    if state.intermediate_states:
        print(f"{indent}Captured States:")
        for key in list(state.intermediate_states.keys())[:5]:  # First 5 keys
            val = state.intermediate_states[key]
            if isinstance(val, np.ndarray):
                print(f"{indent}  - {key}: shape {val.shape}, mean={val.mean():.4f}")
            elif isinstance(val, dict):
                print(f"{indent}  - {key}: {len(val)} items")
            else:
                print(f"{indent}  - {key}: {type(val).__name__}")


def analyze_word(word: str):
    """Analyze a word and show full computational state capture."""
    print(f"\n{'='*60}")
    print(f"Analyzing: '{word}'")
    print(f"{'='*60}\n")

    # Create classifier
    classifier = NounClassifier()

    # Forward pass with state capture
    pred, states = classifier.forward(word, capture_state=True)

    # Prediction
    prediction = "NOUN" if pred > 0.5 else "NOT NOUN"
    confidence = pred if pred > 0.5 else 1 - pred

    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.1%}")
    print()

    # Show captured states
    print(f"Captured {len(states)} computational states:\n")

    for i, state in enumerate(states, 1):
        print(f"[State {i}/{len(states)}]")
        print_state_summary(state)

        # Special handling for attention
        if state.component_type == "SimpleAttentionAtom":
            if 'attention_weights' in state.intermediate_states:
                attn = state.intermediate_states['attention_weights']
                print(f"  Attention Analysis:")
                print(f"    - Attention shape: {attn.shape}")
                print(f"    - Max attention: {attn.max():.4f}")
                print(f"    - Attention entropy: {-(attn * np.log(attn + 1e-9)).sum(axis=-1).mean():.4f}")

                # Show character-level attention
                avg_attn = attn[0].mean(axis=0)  # Average across heads
                char_attn = avg_attn.mean(axis=0)  # Average attention to each position
                print(f"    - Character attention:")
                for j, c in enumerate(word[:len(char_attn)]):
                    if char_attn[j] > 0.05:  # Show significant attention
                        bar = "█" * int(char_attn[j] * 20)
                        print(f"        '{c}': {char_attn[j]:.3f} {bar}")

        print()

    # State size summary
    total_size = sum(s.get_full_state_size() for s in states)
    print(f"Total state captured: {total_size:,} float values")
    print(f"                      ~{total_size * 4 / 1024:.1f} KB")
    print()

    # Save state to JSON for inspection
    state_export = {
        "word": word,
        "prediction": prediction,
        "confidence": float(confidence),
        "num_states": len(states),
        "states": [
            {
                "component_type": s.component_type,
                "semantic_intent": s.semantic_intent,
                "input_shape": s.input_shape,
                "output_shape": s.output_shape,
                "intermediate_state_keys": list(s.intermediate_states.keys())
            }
            for s in states
        ]
    }

    filename = f"state_capture_{word}.json"
    with open(filename, 'w') as f:
        json.dump(state_export, f, indent=2)

    print(f"[OK] Full state exported to: {filename}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExample words to try:")
        print("  Nouns: dog, cat, table, computer, book")
        print("  Non-nouns: quickly, happy, run, very, good")
        sys.exit(1)

    word = sys.argv[1].lower()
    analyze_word(word)
