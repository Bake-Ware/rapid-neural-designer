using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace RapidNeuralDesigner
{
    /// <summary>
    /// Captures complete computational state during neural network forward passes.
    /// This is the C# equivalent of Python's ComputationalState dataclass.
    /// </summary>
    public class ComputationalState
    {
        public string ComponentType { get; set; }
        public string ComponentId { get; set; }
        public DateTime Timestamp { get; set; }

        // Input/Output state
        public Tensor InputArray { get; set; }
        public Tensor OutputArray { get; set; }
        public long[] InputShape { get; set; }
        public long[] OutputShape { get; set; }

        // Internal computational state
        public Dictionary<string, object> IntermediateStates { get; set; }
        public Dictionary<string, Tensor> ParameterStates { get; set; }

        // Semantic metadata
        public string SemanticIntent { get; set; }
        public List<string> ComputationalTrajectory { get; set; }
        public Tensor AttentionPatterns { get; set; }
        public Dictionary<string, object> InformationFlow { get; set; }

        public ComputationalState()
        {
            IntermediateStates = new Dictionary<string, object>();
            ParameterStates = new Dictionary<string, Tensor>();
            ComputationalTrajectory = new List<string>();
            InformationFlow = new Dictionary<string, object>();
            Timestamp = DateTime.Now;
        }

        public long GetFullStateSize()
        {
            long size = InputArray.NumberOfElements + OutputArray.NumberOfElements;

            foreach (var tensor in ParameterStates.Values)
            {
                size += tensor.NumberOfElements;
            }

            return size;
        }
    }

    /// <summary>
    /// Simplified linear layer with complete state capture.
    /// Captures weight matrices, biases, input/output statistics, and transformation metrics.
    /// </summary>
    public class SimpleLinearAtom : Module<Tensor, (Tensor, ComputationalState)>
    {
        private readonly Linear _linear;
        private readonly string _componentId;
        private string _semanticIntent;

        public SimpleLinearAtom(long inFeatures, long outFeatures, bool bias = true, string name = null)
            : base(name ?? $"LinearAtom_{Guid.NewGuid():N}")
        {
            _componentId = this.name;
            _linear = Linear(inFeatures, outFeatures, hasBias: bias);
            _semanticIntent = "linear_transformation";

            RegisterComponents();
        }

        public void SetSemanticIntent(string intent)
        {
            _semanticIntent = intent;
        }

        public override (Tensor, ComputationalState) forward(Tensor x)
        {
            // Capture pre-computation state
            var preState = new Dictionary<string, object>
            {
                ["weight_matrix"] = _linear.weight.clone(),
                ["input_stats"] = new Dictionary<string, double>
                {
                    ["mean"] = x.mean().ToDouble(),
                    ["std"] = x.std().ToDouble(),
                    ["min"] = x.min().ToDouble(),
                    ["max"] = x.max().ToDouble()
                }
            };

            if (_linear.bias != null)
            {
                preState["bias_vector"] = _linear.bias.clone();
            }

            // Forward computation: y = xW^T + b
            var output = _linear.forward(x);

            // Capture post-computation state
            var intermediateStates = new Dictionary<string, object>(preState)
            {
                ["output_stats"] = new Dictionary<string, double>
                {
                    ["mean"] = output.mean().ToDouble(),
                    ["std"] = output.std().ToDouble(),
                    ["min"] = output.min().ToDouble(),
                    ["max"] = output.max().ToDouble()
                },
                ["transformation_magnitude"] = (output - x.mean()).norm(dimensions: new[] { -1L }).ToDouble(),
                ["activation_sparsity"] = (output.eq(0.0).to_type(ScalarType.Float32).mean()).ToDouble()
            };

            // Create computational state
            var state = new ComputationalState
            {
                ComponentType = "SimpleLinearAtom",
                ComponentId = _componentId,
                InputArray = x.clone(),
                OutputArray = output.clone(),
                InputShape = x.shape,
                OutputShape = output.shape,
                IntermediateStates = intermediateStates,
                SemanticIntent = _semanticIntent,
                ComputationalTrajectory = new List<string>
                {
                    "input_reception",
                    "weight_application",
                    "bias_addition",
                    "output_generation"
                }
            };

            // Store parameter states
            state.ParameterStates["weight"] = _linear.weight.clone();
            if (_linear.bias != null)
            {
                state.ParameterStates["bias"] = _linear.bias.clone();
            }

            return (output, state);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _linear?.Dispose();
            }
            base.Dispose(disposing);
        }
    }

    /// <summary>
    /// Simplified multi-head attention mechanism with complete state capture.
    /// Captures Q/K/V projections, attention scores, attention weights, and attention pattern analytics.
    /// </summary>
    public class SimpleAttentionAtom : Module<Tensor, (Tensor, ComputationalState)>
    {
        private readonly long _dModel;
        private readonly long _numHeads;
        private readonly long _dK;
        private readonly string _componentId;
        private string _semanticIntent;

        // Projection matrices
        private readonly Parameter _wQ;
        private readonly Parameter _wK;
        private readonly Parameter _wV;
        private readonly Parameter _wO;

        public SimpleAttentionAtom(long dModel, long numHeads, string name = null)
            : base(name ?? $"AttentionAtom_{Guid.NewGuid():N}")
        {
            if (dModel % numHeads != 0)
            {
                throw new ArgumentException($"dModel ({dModel}) must be divisible by numHeads ({numHeads})");
            }

            _dModel = dModel;
            _numHeads = numHeads;
            _dK = dModel / numHeads;
            _componentId = this.name;
            _semanticIntent = "selective_attention_mechanism";

            // Initialize projection matrices
            _wQ = Parameter(randn(dModel, dModel) * 0.1);
            _wK = Parameter(randn(dModel, dModel) * 0.1);
            _wV = Parameter(randn(dModel, dModel) * 0.1);
            _wO = Parameter(randn(dModel, dModel) * 0.1);

            RegisterComponents();
        }

        public void SetSemanticIntent(string intent)
        {
            _semanticIntent = intent;
        }

        public override (Tensor, ComputationalState) forward(Tensor x)
        {
            var batchSize = x.shape[0];
            var seqLen = x.shape[1];

            // Q, K, V Projections
            var Q = x.matmul(_wQ.transpose(0, 1))
                .reshape(batchSize, seqLen, _numHeads, _dK)
                .permute(0, 2, 1, 3);

            var K = x.matmul(_wK.transpose(0, 1))
                .reshape(batchSize, seqLen, _numHeads, _dK)
                .permute(0, 2, 1, 3);

            var V = x.matmul(_wV.transpose(0, 1))
                .reshape(batchSize, seqLen, _numHeads, _dK)
                .permute(0, 2, 1, 3);

            // Attention Computation
            var scores = Q.matmul(K.transpose(-2, -1)) / Math.Sqrt(_dK);

            // Softmax attention weights
            var attentionWeights = functional.softmax(scores, dim: -1);

            var attendedValues = attentionWeights.matmul(V);

            // Multi-head Concatenation
            var concatenated = attendedValues
                .permute(0, 2, 1, 3)
                .reshape(batchSize, seqLen, _dModel);

            var output = concatenated.matmul(_wO.transpose(0, 1));

            // Comprehensive State Capture
            var intermediateStates = new Dictionary<string, object>
            {
                // QKV states - CRITICAL for neural context bus
                ["Q_projections"] = Q.clone(),
                ["K_projections"] = K.clone(),
                ["V_projections"] = V.clone(),

                // Attention computation states
                ["raw_attention_scores"] = scores.clone(),
                ["attention_weights"] = attentionWeights.clone(),
                ["attended_values"] = attendedValues.clone(),

                // Multi-head configuration
                ["head_configurations"] = new Dictionary<string, long>
                {
                    ["num_heads"] = _numHeads,
                    ["d_k"] = _dK,
                    ["head_dim"] = _dK
                },

                // Attention pattern analytics
                ["attention_entropy"] = -(attentionWeights * attentionWeights.log().add(1e-9)).sum(dimensions: new[] { -1L }),
                ["attention_concentration"] = attentionWeights.max(dimension: -1).values,
                ["attention_diversity"] = (attentionWeights > 0.1).sum(dimensions: new[] { -1L })
            };

            // Create computational state
            var state = new ComputationalState
            {
                ComponentType = "SimpleAttentionAtom",
                ComponentId = _componentId,
                InputArray = x.clone(),
                OutputArray = output.clone(),
                InputShape = x.shape,
                OutputShape = output.shape,
                IntermediateStates = intermediateStates,
                AttentionPatterns = attentionWeights.clone(),
                SemanticIntent = _semanticIntent,
                ComputationalTrajectory = new List<string>
                {
                    "input_reception",
                    "qkv_projection",
                    "attention_score_computation",
                    "attention_weight_normalization",
                    "value_aggregation",
                    "multi_head_concatenation",
                    "output_projection",
                    "output_generation"
                }
            };

            // Store parameter states
            state.ParameterStates["W_Q"] = _wQ.clone();
            state.ParameterStates["W_K"] = _wK.clone();
            state.ParameterStates["W_V"] = _wV.clone();
            state.ParameterStates["W_O"] = _wO.clone();

            return (output, state);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _wQ?.Dispose();
                _wK?.Dispose();
                _wV?.Dispose();
                _wO?.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
