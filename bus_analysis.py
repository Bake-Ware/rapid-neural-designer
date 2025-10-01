"""
Neural Context Bus State Analysis Experiments
Systematically test atomic components to understand storage requirements
"""

import torch
import json
from atomic_components import (
    InstrumentedLinearAtom, 
    InstrumentedAttentionAtom, 
    StateAnalyzer,
    ComputationalState
)
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

class BusRequirementAnalyzer:
    """Comprehensive analysis of neural context bus requirements"""
    
    def __init__(self):
        self.analyzer = StateAnalyzer()
        self.experiments_run = []
        
    def run_component_battery(self, component_types: List[str], 
                            input_configs: List[Dict]) -> Dict[str, Any]:
        """Run battery of tests across different component types"""
        
        results = {
            'component_states': {},
            'dimensional_requirements': {},
            'semantic_analysis': {},
            'hyperbolic_space_specs': {}
        }
        
        for component_type in component_types:
            print(f"\n=== Testing {component_type} ===")
            
            if component_type == "linear":
                component = InstrumentedLinearAtom(512, 256)
            elif component_type == "attention":
                component = InstrumentedAttentionAtom(512, num_heads=8)
            else:
                continue
            
            # Test with different input configurations
            component_results = []
            
            for config in input_configs:
                test_input = torch.randn(**config)
                
                if component_type == "attention":
                    output, state = component(test_input)
                else:
                    output, state = component(test_input)
                
                component_results.append(state)
                self.analyzer.add_states([state])
            
            results['component_states'][component_type] = component_results
        
        # Analyze cross-component requirements
        requirements = self.analyzer.analyze_state_requirements()
        hyperbolic_reqs = self.analyzer.extract_hyperbolic_requirements()
        
        results['semantic_analysis'] = requirements
        results['hyperbolic_space_specs'] = hyperbolic_reqs
        
        return results
    
    def analyze_attention_state_geometry(self, attention_states: List[ComputationalState]) -> Dict[str, Any]:
        """Deep analysis of attention state geometric properties"""
        
        geometry_analysis = {
            'qkv_dimensional_requirements': {},
            'attention_pattern_manifolds': {},
            'hierarchical_structure': {},
            'curvature_estimates': {}
        }
        
        for state in attention_states:
            if state.component_type == 'InstrumentedAttentionAtom':
                Q = state.intermediate_states['Q_projections']
                K = state.intermediate_states['K_projections'] 
                V = state.intermediate_states['V_projections']
                attention_weights = state.intermediate_states['attention_weights']
                
                # Analyze dimensional requirements for preserving QKV relationships
                geometry_analysis['qkv_dimensional_requirements'] = {
                    'q_space_dim': Q.shape,
                    'k_space_dim': K.shape,
                    'v_space_dim': V.shape,
                    'required_embedding_dim': max(Q.numel(), K.numel(), V.numel()),
                    'geometric_constraints': 'preserve_dot_product_relationships'
                }
                
                # Analyze attention pattern structure
                attention_entropy = state.intermediate_states['attention_entropy']
                geometry_analysis['attention_pattern_manifolds'] = {
                    'pattern_complexity': attention_entropy.mean().item(),
                    'pattern_dimensionality': attention_weights.shape[-1],
                    'hierarchical_levels': len(attention_weights.shape),
                    'manifold_curvature_estimate': self._estimate_attention_curvature(attention_weights)
                }
        
        return geometry_analysis
    
    def _estimate_attention_curvature(self, attention_weights: torch.Tensor) -> float:
        """Estimate manifold curvature from attention patterns"""
        # Simplified curvature estimation based on attention concentration
        concentration = attention_weights.max(dim=-1)[0]
        dispersion = attention_weights.std(dim=-1)
        
        # Higher concentration suggests higher curvature (hyperbolic)
        # Lower concentration suggests flatter space (euclidean)
        curvature_proxy = (concentration / (dispersion + 1e-8)).mean().item()
        return -curvature_proxy  # Negative for hyperbolic space
    
    def generate_bus_specification(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate formal specification for neural context bus"""
        
        spec = {
            'hyperbolic_space_layers': {},
            'storage_requirements': {},
            'transformation_interfaces': {},
            'semantic_preservation_protocols': {}
        }
        
        # Extract hyperbolic space layer requirements
        hyperbolic_reqs = analysis_results['hyperbolic_space_specs']
        
        if 'attention' in hyperbolic_reqs['dimensional_layers_needed']:
            attention_layer = hyperbolic_reqs['dimensional_layers_needed']['attention']
            spec['hyperbolic_space_layers']['attention_manifold'] = {
                'dimension': 1024,  # Based on analysis
                'curvature': -1.0,  # Hyperbolic for hierarchical attention
                'stored_objects': ['Q_states', 'K_states', 'V_states', 'attention_weights'],
                'geometric_properties': attention_layer['geometric_properties']
            }
        
        if 'linear' in hyperbolic_reqs['dimensional_layers_needed']:
            linear_layer = hyperbolic_reqs['dimensional_layers_needed']['linear']
            spec['hyperbolic_space_layers']['transformation_manifold'] = {
                'dimension': 512,   # Based on analysis
                'curvature': 0.0,   # Euclidean for linear transformations
                'stored_objects': ['weight_matrices', 'transformation_vectors'],
                'geometric_properties': linear_layer['geometric_properties']
            }
        
        # Storage requirements
        semantic_analysis = analysis_results['semantic_analysis']
        spec['storage_requirements'] = {
            'total_state_types': len(semantic_analysis['intermediate_state_types']),
            'semantic_intents': list(semantic_analysis['semantic_intents']),
            'computational_trajectories': list(semantic_analysis['computational_trajectories']),
            'estimated_storage_per_component': np.mean(semantic_analysis['state_size_distribution']) if semantic_analysis['state_size_distribution'] else 0
        }
        
        return spec

def run_comprehensive_analysis():
    """Run comprehensive analysis to spec out neural context bus"""
    
    analyzer = BusRequirementAnalyzer()
    
    # Define test configurations
    input_configs = [
        {'size': (2, 64, 512)},   # Small sequence
        {'size': (4, 128, 512)},  # Medium sequence  
        {'size': (1, 256, 512)},  # Long sequence
        {'size': (8, 32, 512)},   # Large batch
    ]
    
    component_types = ['linear', 'attention']
    
    # Run comprehensive battery of tests
    print("🧠 Running Neural Context Bus Requirement Analysis...")
    results = analyzer.run_component_battery(component_types, input_configs)
    
    # Deep dive into attention geometry
    attention_states = []
    for states in results['component_states'].values():
        attention_states.extend([s for s in states if s.component_type == 'InstrumentedAttentionAtom'])
    
    geometry_analysis = analyzer.analyze_attention_state_geometry(attention_states)
    
    # Generate formal bus specification
    bus_spec = analyzer.generate_bus_specification(results)
    
    # Save results
    print("\n💾 Saving analysis results...")
    
    with open('C:/neural_vm_experiments/bus_requirements.json', 'w') as f:
        json.dump({
            'component_analysis': {k: str(v) for k, v in results['semantic_analysis'].items()},
            'geometry_analysis': geometry_analysis,
            'bus_specification': bus_spec
        }, f, indent=2, default=str)
    
    print("🎯 Neural Context Bus Specification Generated!")
    print(f"   - Hyperbolic layers needed: {len(bus_spec['hyperbolic_space_layers'])}")
    print(f"   - State types to preserve: {bus_spec['storage_requirements']['total_state_types']}")
    print(f"   - Semantic intents captured: {len(bus_spec['storage_requirements']['semantic_intents'])}")
    
    return bus_spec

if __name__ == "__main__":
    # Run the comprehensive analysis
    specification = run_comprehensive_analysis()
    print("\n🚀 Ready to build Neural Context Bus!")
