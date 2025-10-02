"""
Train a simple noun classifier using atomic components with full state capture.
Demonstrates RND's capability to capture complete computational state during training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Import atomic components from neuralAtomLib
from neuralAtomLib import (
    SimpleLinearAtom,
    SimpleAttentionAtom,
    ComputationalState
)


@dataclass
class CharacterEmbedding:
    """Simple character-level embedding for words."""
    char_to_idx: Dict[str, int]
    embedding_dim: int
    max_word_len: int

    def encode(self, word: str) -> np.ndarray:
        """Encode a word as character indices, padded to max_word_len."""
        word = word.lower()[:self.max_word_len]
        indices = [self.char_to_idx.get(c, 0) for c in word]
        # Pad with zeros
        indices += [0] * (self.max_word_len - len(indices))

        # Convert to one-hot encoding
        encoded = np.zeros((self.max_word_len, self.embedding_dim))
        for i, idx in enumerate(indices):
            if idx < self.embedding_dim:
                encoded[i, idx] = 1.0

        return encoded


class NounClassifier:
    """Simple noun classifier using atomic components."""

    def __init__(self, embedding_dim: int = 32, max_word_len: int = 20, hidden_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.max_word_len = max_word_len
        self.hidden_dim = hidden_dim

        # Create character vocabulary (a-z + space/padding)
        chars = ' abcdefghijklmnopqrstuvwxyz'
        self.vocab_size = len(chars)  # 27 chars
        # Round up to nearest multiple of num_heads (4) -> 28
        self.d_model = ((self.vocab_size + 3) // 4) * 4

        self.char_encoder = CharacterEmbedding(
            char_to_idx={c: i for i, c in enumerate(chars)},
            embedding_dim=self.d_model,
            max_word_len=max_word_len
        )

        # Atomic components
        self.attention = SimpleAttentionAtom(
            d_model=self.d_model,
            num_heads=4
        )
        self.attention.semantic_intent = "character_attention"

        self.projection = SimpleLinearAtom(
            in_features=self.d_model,
            out_features=hidden_dim
        )
        self.projection.semantic_intent = "feature_extraction"

        self.classifier = SimpleLinearAtom(
            in_features=hidden_dim,
            out_features=1
        )
        self.classifier.semantic_intent = "binary_classification"

    def forward(self, word: str, capture_state: bool = True) -> Tuple[float, List[ComputationalState]]:
        """Forward pass through the classifier."""
        states = []

        # Encode word
        x = self.char_encoder.encode(word)  # (max_word_len, embedding_dim)
        x = x.reshape(1, self.max_word_len, -1)  # (1, seq_len, embed_dim)

        # Attention over characters
        attn_out, attn_state = self.attention.forward(x)
        if capture_state:
            states.append(attn_state)

        # Pool attention output (mean over sequence)
        pooled = np.mean(attn_out, axis=1)  # (1, embed_dim)

        # Project to hidden dimension
        projected, proj_state = self.projection.forward(pooled)
        if capture_state:
            states.append(proj_state)

        # ReLU activation
        activated = np.maximum(0, projected)

        # Classify
        logit, class_state = self.classifier.forward(activated)
        if capture_state:
            states.append(class_state)

        # Sigmoid for binary classification
        prob = 1 / (1 + np.exp(-logit[0, 0]))

        return prob, states

    def train_step(self, word: str, label: bool, learning_rate: float = 0.01) -> Tuple[float, List[ComputationalState]]:
        """Single training step with gradient descent."""
        # Forward pass
        pred, states = self.forward(word, capture_state=True)

        # Binary cross-entropy loss
        target = 1.0 if label else 0.0
        loss = -(target * np.log(pred + 1e-10) + (1 - target) * np.log(1 - pred + 1e-10))

        # Compute gradient
        grad = pred - target

        # Backward pass (simplified gradient descent on final layer only)
        # Get the input to classifier from states
        class_state = states[-1]
        input_to_classifier = class_state.input_array

        # Simplified weight update: w = w - lr * grad * input
        # Proper dimensions: (out_features, in_features) -= scalar * (1, in_features)
        if hasattr(self.classifier, 'weight'):
            grad_w = grad * input_to_classifier  # (1, hidden_dim)
            self.classifier.weight -= learning_rate * grad_w * 0.01

        return loss, states


def load_dataset(filepath: str = "noun_dataset.json") -> List[Tuple[str, bool]]:
    """Load the noun classification dataset."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    return [(item['word'], item['is_noun']) for item in data['words']]


def train(num_epochs: int = 10, learning_rate: float = 0.01):
    """Train the noun classifier."""
    print("Loading dataset...")
    dataset = load_dataset()

    print(f"Training on {len(dataset)} words...")
    print(f"Nouns: {sum(1 for _, label in dataset if label)}")
    print(f"Non-nouns: {sum(1 for _, label in dataset if not label)}")
    print()

    # Initialize classifier
    classifier = NounClassifier()

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0

        # Shuffle dataset
        np.random.shuffle(dataset)

        for i, (word, label) in enumerate(dataset):
            loss, states = classifier.train_step(word, label, learning_rate)
            total_loss += loss

            # Check accuracy
            pred, _ = classifier.forward(word, capture_state=False)
            if (pred > 0.5) == label:
                correct += 1

            # Print progress
            if (i + 1) % 200 == 0:
                avg_loss = total_loss / (i + 1)
                acc = correct / (i + 1)
                print(f"Epoch {epoch+1}/{num_epochs} | Step {i+1}/{len(dataset)} | "
                      f"Loss: {avg_loss:.4f} | Acc: {acc:.3f}")

        avg_loss = total_loss / len(dataset)
        accuracy = correct / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} Complete | Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy:.3f}")
        print()

    # Save model
    print("Training complete! Saving model...")
    np.savez('noun_classifier.npz',
             attention_wq=classifier.attention.w_q,
             attention_wk=classifier.attention.w_k,
             attention_wv=classifier.attention.w_v,
             attention_wo=classifier.attention.w_o,
             projection_weight=classifier.projection.weight,
             projection_bias=classifier.projection.bias,
             classifier_weight=classifier.classifier.weight,
             classifier_bias=classifier.classifier.bias)

    return classifier


def test_classifier(classifier: NounClassifier, test_words: List[str]):
    """Test the classifier on example words."""
    print("\n" + "="*50)
    print("Testing Classifier")
    print("="*50)

    for word in test_words:
        pred, states = classifier.forward(word, capture_state=True)
        prediction = "NOUN" if pred > 0.5 else "NOT NOUN"
        confidence = pred if pred > 0.5 else 1 - pred

        print(f"\nWord: '{word}'")
        print(f"Prediction: {prediction} (confidence: {confidence:.3f})")

        # Show attention patterns
        attn_state = states[0]
        if 'attention_weights' in attn_state.intermediate_states:
            attn_weights = attn_state.intermediate_states['attention_weights']
            # Average attention across heads
            avg_attn = np.mean(attn_weights[0], axis=0)  # (seq_len, seq_len)
            # Get attention on each character position
            char_attn = np.mean(avg_attn, axis=0)  # Average attention to each position

            print(f"Character attention: ", end="")
            for i, c in enumerate(word[:len(char_attn)]):
                if char_attn[i] > 0.1:  # Only show significant attention
                    print(f"{c}({char_attn[i]:.2f}) ", end="")
            print()


if __name__ == "__main__":
    # Train the model
    classifier = train(num_epochs=5, learning_rate=0.01)

    # Test on example words
    test_words = ["dog", "cat", "quickly", "table", "run", "happy", "computer", "sing", "book", "very"]
    test_classifier(classifier, test_words)
