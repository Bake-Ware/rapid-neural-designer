"""
Quick test runner for neural VM experiments
Run this to immediately start capturing atomic component states
"""

import torch
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from atomic_components import InstrumentedLinearAtom, InstrumentedAttentionAtom, StateAnalyzer
    print("✅ Successfully imported atomic components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have PyTorch installed: pip install torch")
    sys.exit(1)

def quick_test():
    """Quick test to verify everything works"""
    print("\n🧪 Running quick test of atomic components...")
    
    # Create test components
    linear = InstrumentedLinearAtom(128, 64)
    attention = InstrumentedAttentionAtom(128, num_heads=4)
    
    # Create test input
    test_input = torch.randn(2, 32, 128)
    print(f"Test input shape: {test_input.shape}")
    
    # Test linear component
    print("\n📊 Testing Linear Atom...")
    linear_output, linear_state = linear(test_input)
    print(f"Linear output shape: {linear_output.shape}")
    print(f"Linear state captured: {len(linear_state.intermediate_states)} intermediate states")
    print(f"Semantic intent: {linear_state.semantic_intent}")
    
    # Test attention component  
    print("\n🎯 Testing Attention Atom...")
    attention_output, attention_state = attention(test_input)
    print(f"Attention output shape: {attention_output.shape}")
    print(f"Attention state captured: {len(attention_state.intermediate_states)} intermediate states")
    print(f"QKV states preserved: {'Q_projections' in attention_state.intermediate_states}")
    print(f"Attention patterns captured: {attention_state.attention_patterns is not None}")
    
    # Quick analysis
    analyzer = StateAnalyzer()
    analyzer.add_states([linear_state, attention_state])
    requirements = analyzer.analyze_state_requirements()
    
    print(f"\n📋 Analysis Results:")
    print(f"Total states captured: {requirements['total_states_captured']}")
    print(f"Unique state types: {len(requirements['intermediate_state_types'])}")
    print(f"Semantic intents: {list(requirements['semantic_intents'])}")
    
    print("\n✅ Quick test completed successfully!")
    print("🚀 Ready to run full bus_analysis.py for complete specifications")

if __name__ == "__main__":
    quick_test()
