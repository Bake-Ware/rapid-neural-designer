"""
Instrumented Atomic Neural Components
Captures ALL internal state for neural context bus design
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict

@dataclass
class ComputationalState:
    """Captures complete computational state during forward pass"""
    component_type: str
    component_id: str
    timestamp: float
    
    # Input/Output state
    input_tensor: torch.Tensor
    output_tensor: torch.Tensor
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    
    # Internal computational state
    intermediate_states: Dict[str, torch.Tensor] = field(default_factory=dict)
    parameter_states: Dict[str, torch.Tensor] = field(default_factory=dict)
    gradient_states: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # Semantic metadata
    semantic_intent: str = ""
    computational_trajectory: List[str] = field(default_factory=list)
    attention_patterns: Optional[torch.Tensor] = None
    information_flow: Dict[str, Any] = field(default_factory=dict)
    
    # Control flow state
    gating_decisions: Dict[str, torch.Tensor] = field(default_factory=dict)
    routing_choices: Dict[str, Any] = field(default_factory=dict)
    
    def get_full_state_size(self) -> int:
        """Calculate total state information captured"""
        size = self.input_tensor.numel() + self.output_tensor.numel()
        for tensor in self.intermediate_states.values():
            size += tensor.numel()
        for tensor in self.parameter_states.values():
            size += tensor.numel()
        return size

class StateInstrumentation:
    """Mixin for capturing full computational state"""
    
    def __init__(self):
        self.state_history: List[ComputationalState] = []
        self.capture_enabled = True
        self.component_id = f"{self.__class__.__name__}_{id(self)}"
    
    def capture_state(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor, 
                     intermediate_states: Dict[str, torch.Tensor], **kwargs) -> ComputationalState:
        """Capture complete computational state"""
        if not self.capture_enabled:
            return None
            
        state = ComputationalState(
            component_type=self.__class__.__name__,
            component_id=self.component_id,
            timestamp=torch.cuda.Event().record().elapsed_time(torch.cuda.Event()) if torch.cuda.is_available() else 0.0,
            input_tensor=input_tensor.clone().detach(),
            output_tensor=output_tensor.clone().detach(),
            input_shape=input_tensor.shape,
            output_shape=output_tensor.shape,
            intermediate_states={k: v.clone().detach() for k, v in intermediate_states.items()},
            **kwargs
        )
        
        self.state_history.append(state)
        return state

class InstrumentedLinearAtom(nn.Module, StateInstrumentation):
    """Linear layer with complete state capture"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        nn.Module.__init__(self)
        StateInstrumentation.__init__(self)
        
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias)
        
        # Semantic annotation
        self.semantic_intent = "linear_transformation"
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ComputationalState]:
        # Capture pre-computation state
        pre_state = {
            'weight_matrix': self.linear.weight.clone().detach(),
            'input_stats': {
                'mean': x.mean().item(),
                'std': x.std().item(),
                'min': x.min().item(),
                'max': x.max().item()
            }
        }
        
        if self.linear.bias is not None:
            pre_state['bias_vector'] = self.linear.bias.clone().detach()
        
        # Forward computation
        output = self.linear(x)
        
        # Capture post-computation state
        intermediate_states = {
            **pre_state,
            'output_stats': {
                'mean': output.mean().item(),
                'std': output.std().item(),
                'min': output.min().item(),
                'max': output.max().item()
            },
            'transformation_magnitude': torch.norm(output - x.mean(), dim=-1),
            'activation_sparsity': (output == 0).float().mean().item()
        }
        
        # Capture complete state
        state = self.capture_state(
            input_tensor=x,
            output_tensor=output,
            intermediate_states=intermediate_states,
            semantic_intent=self.semantic_intent,
            computational_trajectory=['input_projection', 'bias_addition', 'output_generation']
        )
        
        return output, state

class InstrumentedAttentionAtom(nn.Module, StateInstrumentation):
    """Attention mechanism with complete state capture"""
    
    def __init__(self, d_model: int, num_heads: int = 8):
        nn.Module.__init__(self)
        StateInstrumentation.__init__(self)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model) 
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.semantic_intent = "selective_attention_mechanism"
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ComputationalState]:
        batch_size, seq_len, _ = x.shape
        
        # === Q, K, V Projections ===
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # === Attention Computation ===
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, V)
        
        # === Multi-head Concatenation ===
        concatenated = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(concatenated)
        
        # === COMPREHENSIVE STATE CAPTURE ===
        intermediate_states = {
            # QKV states - CRITICAL for neural context bus
            'Q_projections': Q.clone().detach(),
            'K_projections': K.clone().detach(),
            'V_projections': V.clone().detach(),
            
            # Attention computation states
            'raw_attention_scores': scores.clone().detach(),
            'attention_weights': attention_weights.clone().detach(),
            'attended_values': attended_values.clone().detach(),
            
            # Multi-head configuration
            'head_configurations': {
                'num_heads': self.num_heads,
                'd_k': self.d_k,
                'head_dim': self.d_k
            },
            
            # Attention pattern analytics
            'attention_entropy': -torch.sum(attention_weights * torch.log(attention_weights + 1e-9), dim=-1),
            'attention_concentration': attention_weights.max(dim=-1)[0],
            'attention_diversity': (attention_weights > 0.1).sum(dim=-1).float(),
            
            # Information flow metrics
            'query_key_similarity': torch.cosine_similarity(Q.flatten(2), K.flatten(2), dim=-1),
            'value_utilization': torch.norm(attended_values, dim=-1),
            
            # Transformation magnitudes
            'q_transform_magnitude': torch.norm(Q, dim=-1),
            'k_transform_magnitude': torch.norm(K, dim=-1),
            'v_transform_magnitude': torch.norm(V, dim=-1),
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
            'dominant_attention_heads': attention_weights.mean(dim=(0, 2)).argmax(dim=-1),
            'attention_pattern_type': self._classify_attention_pattern(attention_weights),
            'information_bottleneck': self._compute_information_bottleneck(Q, K, V, attention_weights)
        }
        
        state = self.capture_state(
            input_tensor=x,
            output_tensor=output,
            intermediate_states=intermediate_states,
            semantic_intent=self.semantic_intent,
            computational_trajectory=computational_trajectory,
            attention_patterns=attention_weights,
            information_flow=attention_metadata
        )
        
        return output, state
    
    def _classify_attention_pattern(self, attention_weights: torch.Tensor) -> str:
        """Classify the type of attention pattern"""
        # Simple heuristics - could be much more sophisticated
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-9), dim=-1)
        avg_entropy = entropy.mean()
        
        if avg_entropy < 1.0:
            return "focused_attention"
        elif avg_entropy > 3.0:
            return "distributed_attention" 
        else:
            return "balanced_attention"
    
    def _compute_information_bottleneck(self, Q, K, V, attention_weights) -> Dict[str, float]:
        """Compute information flow metrics"""
        # Simplified information theory metrics
        return {
            'mutual_information': torch.sum(attention_weights * torch.log(attention_weights + 1e-9)).item(),
            'effective_attention_heads': (attention_weights.mean(dim=(0, 2, 3)) > 0.1).sum().item(),
            'attention_sparsity': (attention_weights < 0.01).float().mean().item()
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
    
    def extract_hyperbolic_requirements(self) -> Dict[str, Any]:
        """Extract requirements for hyperbolic space design"""
        
        requirements = {
            'dimensional_layers_needed': {},
            'curvature_requirements': {},
            'embedding_dimensions': {},
            'state_preservation_priorities': {}
        }
        
        # Analyze which states need which type of geometric preservation
        for state in self.state_database:
            if state.component_type == 'InstrumentedAttentionAtom':
                # Attention states need hyperbolic space for hierarchical relationships
                requirements['dimensional_layers_needed']['attention'] = {
                    'qkv_states': state.intermediate_states['Q_projections'].shape,
                    'attention_weights': state.intermediate_states['attention_weights'].shape,
                    'geometric_properties': 'hierarchical_attention_patterns'
                }
            
            elif state.component_type == 'InstrumentedLinearAtom':
                # Linear states need euclidean-like space for transformation preservation
                requirements['dimensional_layers_needed']['linear'] = {
                    'weight_matrices': state.intermediate_states['weight_matrix'].shape,
                    'transformation_vectors': state.output_tensor.shape,
                    'geometric_properties': 'linear_transformations'
                }
        
        return requirements

# Example usage - building the atomic components to spec out bus requirements
def run_state_capture_experiment():
    """Run experiment to capture all neural states"""
    
    # Initialize components
    linear_atom = InstrumentedLinearAtom(512, 256)
    attention_atom = InstrumentedAttentionAtom(512, num_heads=8)
    analyzer = StateAnalyzer()
    
    # Generate test data
    batch_size, seq_len, d_model = 2, 64, 512
    test_input = torch.randn(batch_size, seq_len, d_model)
    
    # Capture states from linear transformation
    linear_output, linear_state = linear_atom(test_input)
    
    # Capture states from attention mechanism  
    attention_output, attention_state = attention_atom(test_input)
    
    # Analyze what the neural context bus needs to handle
    analyzer.add_states([linear_state, attention_state])
    requirements = analyzer.analyze_state_requirements()
    hyperbolic_reqs = analyzer.extract_hyperbolic_requirements()
    
    print("=== NEURAL CONTEXT BUS REQUIREMENTS ===")
    print(f"State types to preserve: {requirements['intermediate_state_types']}")
    print(f"Semantic intents: {requirements['semantic_intents']}")
    print(f"Computational trajectories: {len(requirements['computational_trajectories'])}")
    print(f"Hyperbolic space requirements: {hyperbolic_reqs['dimensional_layers_needed'].keys()}")
    
    return requirements, hyperbolic_reqs

if __name__ == "__main__":
    requirements, hyperbolic_reqs = run_state_capture_experiment()
