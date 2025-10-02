"""
Simplified Neural VM State Capture Experiment
Using numpy instead of PyTorch to demonstrate core concepts
"""

import numpy as np
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import time

@dataclass
class ComputationalState:
    """Captures complete computational state during forward pass"""
    component_type: str
    component_id: str
    timestamp: float
    
    # Input/Output state
    input_array: np.ndarray
    output_array: np.ndarray
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    
    # Internal computational state
    intermediate_states: Dict[str, np.ndarray] = field(default_factory=dict)
    parameter_states: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Semantic metadata
    semantic_intent: str = ""
    computational_trajectory: List[str] = field(default_factory=list)
    attention_patterns: Optional[np.ndarray] = None
    information_flow: Dict[str, Any] = field(default_factory=dict)
    
    def get_full_state_size(self) -> int:
        """Calculate total state information captured"""
        size = self.input_array.size + self.output_array.size
        for array in self.intermediate_states.values():
            if isinstance(array, np.ndarray):
                size += array.size
        for array in self.parameter_states.values():
            if isinstance(array, np.ndarray):
                size += array.size
        return size

class SimpleLinearAtom:
    """Simplified linear layer with state capture"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.component_id = f"LinearAtom_{id(self)}"
        
        # Initialize parameters
        self.weight = np.random.randn(out_features, in_features) * 0.1
        self.bias = np.random.randn(out_features) * 0.1 if bias else None
        
        self.semantic_intent = "linear_transformation"
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, ComputationalState]:
        # Capture pre-computation state
        pre_state = {
            'weight_matrix': self.weight.copy(),
            'input_stats': {
                'mean': float(x.mean()),
                'std': float(x.std()),
                'min': float(x.min()),
                'max': float(x.max())
            }
        }
        
        if self.bias is not None:
            pre_state['bias_vector'] = self.bias.copy()
        
        # Forward computation: y = xW^T + b
        output = np.dot(x, self.weight.T)
        if self.bias is not None:
            output = output + self.bias
        
        # Capture post-computation state
        intermediate_states = {
            **pre_state,
            'output_stats': {
                'mean': float(output.mean()),
                'std': float(output.std()),
                'min': float(output.min()),
                'max': float(output.max())
            },
            'transformation_magnitude': np.linalg.norm(output - x.mean(), axis=-1),
            'activation_sparsity': float((output == 0).mean())
        }
        
        # Create computational state
        state = ComputationalState(
            component_type="SimpleLinearAtom",
            component_id=self.component_id,
            timestamp=time.time(),
            input_array=x.copy(),
            output_array=output.copy(),
            input_shape=x.shape,
            output_shape=output.shape,
            intermediate_states=intermediate_states,
            semantic_intent=self.semantic_intent,
            computational_trajectory=['input_projection', 'bias_addition', 'output_generation']
        )
        
        return output, state

class SimpleAttentionAtom:
    """Simplified attention mechanism with state capture"""
    
    def __init__(self, d_model: int, num_heads: int = 8):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.component_id = f"AttentionAtom_{id(self)}"
        
        # Initialize projection matrices
        self.w_q = np.random.randn(d_model, d_model) * 0.1
        self.w_k = np.random.randn(d_model, d_model) * 0.1
        self.w_v = np.random.randn(d_model, d_model) * 0.1
        self.w_o = np.random.randn(d_model, d_model) * 0.1
        
        self.semantic_intent = "selective_attention_mechanism"
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ComputationalState]:
        batch_size, seq_len, _ = x.shape
        
        # === Q, K, V Projections ===
        Q = np.dot(x, self.w_q.T).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = np.dot(x, self.w_k.T).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = np.dot(x, self.w_v.T).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # === Attention Computation ===
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Softmax attention weights
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        attended_values = np.matmul(attention_weights, V)
        
        # === Multi-head Concatenation ===
        concatenated = attended_values.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        output = np.dot(concatenated, self.w_o.T)
        
        # === COMPREHENSIVE STATE CAPTURE ===
        intermediate_states = {
            # QKV states - CRITICAL for neural context bus
            'Q_projections': Q.copy(),
            'K_projections': K.copy(),
            'V_projections': V.copy(),
            
            # Attention computation states
            'raw_attention_scores': scores.copy(),
            'attention_weights': attention_weights.copy(),
            'attended_values': attended_values.copy(),
            
            # Multi-head configuration
            'head_configurations': {
                'num_heads': self.num_heads,
                'd_k': self.d_k,
                'head_dim': self.d_k
            },
            
            # Attention pattern analytics
            'attention_entropy': -np.sum(attention_weights * np.log(attention_weights + 1e-9), axis=-1),
            'attention_concentration': np.max(attention_weights, axis=-1),
            'attention_diversity': np.sum(attention_weights > 0.1, axis=-1),
            
            # Information flow metrics
            'query_key_similarity': np.sum(Q.reshape(batch_size, self.num_heads, -1) * 
                                         K.reshape(batch_size, self.num_heads, -1), axis=-1) / (
                                        np.linalg.norm(Q.reshape(batch_size, self.num_heads, -1), axis=-1) * 
                                        np.linalg.norm(K.reshape(batch_size, self.num_heads, -1), axis=-1) + 1e-8),
            
            # Transformation magnitudes
            'q_transform_magnitude': np.linalg.norm(Q, axis=-1),
            'k_transform_magnitude': np.linalg.norm(K, axis=-1),
            'v_transform_magnitude': np.linalg.norm(V, axis=-1),
        }
        
        # Semantic trajectory tracking
        computational_trajectory = [
            'input_reception',
            'qkv_projection', 
            'attention_score_computation',
            'attention_weight_normalization',
            'value_aggregation',
            'multi_head_concatenation',
            'output_projection'
        ]
        
        # Attention-specific metadata
        attention_metadata = {
            'dominant_attention_heads': np.argmax(attention_weights.mean(axis=(0, 2)), axis=-1),
            'attention_pattern_type': self._classify_attention_pattern(attention_weights),
            'information_bottleneck': self._compute_information_bottleneck(Q, K, V, attention_weights)
        }
        
        state = ComputationalState(
            component_type="SimpleAttentionAtom",
            component_id=self.component_id,
            timestamp=time.time(),
            input_array=x.copy(),
            output_array=output.copy(),
            input_shape=x.shape,
            output_shape=output.shape,
            intermediate_states=intermediate_states,
            semantic_intent=self.semantic_intent,
            computational_trajectory=computational_trajectory,
            attention_patterns=attention_weights,
            information_flow=attention_metadata
        )
        
        return output, state
    
    def _classify_attention_pattern(self, attention_weights: np.ndarray) -> str:
        """Classify the type of attention pattern"""
        entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-9), axis=-1)
        avg_entropy = entropy.mean()
        
        if avg_entropy < 1.0:
            return "focused_attention"
        elif avg_entropy > 3.0:
            return "distributed_attention" 
        else:
            return "balanced_attention"
    
    def _compute_information_bottleneck(self, Q, K, V, attention_weights) -> Dict[str, float]:
        """Compute information flow metrics"""
        return {
            'mutual_information': float(np.sum(attention_weights * np.log(attention_weights + 1e-9))),
            'effective_attention_heads': float(np.sum(attention_weights.mean(axis=(0, 2, 3)) > 0.1)),
            'attention_sparsity': float((attention_weights < 0.01).mean())
        }

class StateAnalyzer:
    """Analyze captured states to understand bus requirements"""
    
    def __init__(self):
        self.state_database = []
    
    def add_states(self, states: List[ComputationalState]):
        self.state_database.extend(states)
    
    def analyze_state_requirements(self) -> Dict[str, Any]:
        """Analyze what the neural context bus needs to store"""
        
        analysis = {
            'total_states_captured': len(self.state_database),
            'state_size_distribution': [],
            'semantic_intents': set(),
            'computational_trajectories': set(),
            'intermediate_state_types': set(),
            'attention_pattern_types': set(),
        }
        
        for state in self.state_database:
            analysis['state_size_distribution'].append(state.get_full_state_size())
            analysis['semantic_intents'].add(state.semantic_intent)
            analysis['computational_trajectories'].update(state.computational_trajectory)
            analysis['intermediate_state_types'].update(state.intermediate_states.keys())
            
            if state.attention_patterns is not None:
                # Analyze attention patterns for hyperbolic space requirements
                pass
        
        return analysis

def run_neural_vm_experiment():
    """Run the complete neural VM state capture experiment"""
    
    print("Neural VM State Capture Experiment")
    print("=" * 50)
    
    # Initialize components
    print("\nInitializing atomic components...")
    linear_atom = SimpleLinearAtom(512, 256)
    attention_atom = SimpleAttentionAtom(512, num_heads=8)
    analyzer = StateAnalyzer()
    
    print(f"[OK] Linear Atom: {linear_atom.in_features} -> {linear_atom.out_features}")
    print(f"[OK] Attention Atom: {attention_atom.d_model}d, {attention_atom.num_heads} heads")
    
    # Test configurations
    test_configs = [
        (2, 64, 512),   # Small sequence
        (4, 32, 512),   # Medium batch
        (1, 128, 512),  # Long sequence
    ]
    
    all_states = []
    
    for i, (batch_size, seq_len, d_model) in enumerate(test_configs):
        print(f"\nTest {i+1}: Input shape ({batch_size}, {seq_len}, {d_model})")
        
        # Generate test data
        test_input = np.random.randn(batch_size, seq_len, d_model)
        
        # Test linear component
        print("   [LINEAR] Testing Linear Atom...")
        linear_output, linear_state = linear_atom.forward(test_input)
        print(f"      Output shape: {linear_output.shape}")
        print(f"      States captured: {len(linear_state.intermediate_states)}")
        print(f"      State size: {linear_state.get_full_state_size():,} elements")
        
        # Test attention component  
        print("   [ATTENTION] Testing Attention Atom...")
        attention_output, attention_state = attention_atom.forward(test_input)
        print(f"      Output shape: {attention_output.shape}")
        print(f"      States captured: {len(attention_state.intermediate_states)}")
        print(f"      State size: {attention_state.get_full_state_size():,} elements")
        print(f"      QKV preserved: {'Q_projections' in attention_state.intermediate_states}")
        print(f"      Attention pattern: {attention_state.information_flow.get('attention_pattern_type', 'unknown')}")
        
        all_states.extend([linear_state, attention_state])
        analyzer.add_states([linear_state, attention_state])
    
    # Comprehensive analysis
    print(f"\nCOMPREHENSIVE ANALYSIS")
    print("=" * 50)
    
    requirements = analyzer.analyze_state_requirements()
    
    print(f"Total states captured: {requirements['total_states_captured']}")
    print(f"Average state size: {np.mean(requirements['state_size_distribution']):.0f} elements")
    print(f"Max state size: {max(requirements['state_size_distribution']):,} elements")
    
    print(f"\nSemantic Intents Captured:")
    for intent in requirements['semantic_intents']:
        print(f"   * {intent}")
    
    print(f"\nComputational Trajectories ({len(requirements['computational_trajectories'])}):")
    for i, trajectory in enumerate(sorted(requirements['computational_trajectories'])):
        print(f"   {i+1}. {trajectory}")
    
    print(f"\nIntermediate State Types ({len(requirements['intermediate_state_types'])}):")
    for i, state_type in enumerate(sorted(requirements['intermediate_state_types'])):
        print(f"   {i+1}. {state_type}")
    
    # Generate neural context bus requirements
    print(f"\nNEURAL CONTEXT BUS REQUIREMENTS")
    print("=" * 50)
    
    bus_spec = {
        'hyperbolic_space_layers': {
            'attention_manifold': {
                'dimension': 1024,
                'curvature': -1.0,  # Hyperbolic for hierarchical attention
                'stored_objects': ['Q_states', 'K_states', 'V_states', 'attention_weights'],
                'geometric_properties': 'hierarchical_attention_patterns'
            },
            'transformation_manifold': {
                'dimension': 512,
                'curvature': 0.0,   # Euclidean for linear transformations
                'stored_objects': ['weight_matrices', 'transformation_vectors'],
                'geometric_properties': 'linear_transformations'
            },
            'semantic_manifold': {
                'dimension': 256,
                'curvature': -0.5,  # Mild hyperbolic for semantic relationships
                'stored_objects': ['semantic_intents', 'computational_trajectories'],
                'geometric_properties': 'concept_hierarchies'
            }
        },
        'storage_requirements': {
            'total_state_types': len(requirements['intermediate_state_types']),
            'semantic_intents': list(requirements['semantic_intents']),
            'computational_trajectories': list(requirements['computational_trajectories']),
            'estimated_storage_per_component': int(np.mean(requirements['state_size_distribution']))
        },
        'interface_requirements': {
            'read_operations': ['query_by_semantic_intent', 'query_by_trajectory', 'query_by_state_type'],
            'write_operations': ['store_qkv_states', 'store_attention_patterns', 'store_transformations'],
            'transform_operations': ['attention_to_convolution', 'linear_to_graph', 'semantic_translation']
        }
    }
    
    print("Hyperbolic Space Layers:")
    for layer_name, layer_spec in bus_spec['hyperbolic_space_layers'].items():
        print(f"   * {layer_name}:")
        print(f"     - Dimension: {layer_spec['dimension']}")
        print(f"     - Curvature: {layer_spec['curvature']}")
        print(f"     - Objects: {len(layer_spec['stored_objects'])}")
    
    print(f"\nStorage Capacity:")
    print(f"   * State types: {bus_spec['storage_requirements']['total_state_types']}")
    print(f"   * Avg component size: {bus_spec['storage_requirements']['estimated_storage_per_component']:,} elements")
    print(f"   * Semantic intents: {len(bus_spec['storage_requirements']['semantic_intents'])}")
    
    print(f"\nInterface Operations:")
    print(f"   * Read ops: {len(bus_spec['interface_requirements']['read_operations'])}")
    print(f"   * Write ops: {len(bus_spec['interface_requirements']['write_operations'])}")
    print(f"   * Transform ops: {len(bus_spec['interface_requirements']['transform_operations'])}")
    
    # Save results
    print(f"\nSaving results to bus_requirements.json...")
    
    # Convert sets to lists for JSON serialization
    serializable_requirements = {
        'total_states_captured': requirements['total_states_captured'],
        'state_size_distribution': requirements['state_size_distribution'],
        'semantic_intents': list(requirements['semantic_intents']),
        'computational_trajectories': list(requirements['computational_trajectories']),
        'intermediate_state_types': list(requirements['intermediate_state_types']),
    }
    
    with open('C:/neural_vm_experiments/bus_requirements.json', 'w') as f:
        json.dump({
            'experiment_results': serializable_requirements,
            'bus_specification': bus_spec,
            'experiment_timestamp': time.time()
        }, f, indent=2)
    
    print(f"\n[SUCCESS] EXPERIMENT COMPLETE!")
    print(f"[SPECS] Neural Context Bus specifications generated")
    print(f"[FILE] Results saved to: C:/neural_vm_experiments/bus_requirements.json")
    print(f"\n[NEXT] Ready for Phase 2: Build the Neural Context Bus!")
    
    return bus_spec

if __name__ == "__main__":
    bus_spec = run_neural_vm_experiment()
