using System;
using Unity.InferenceEngine;
using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// One exported brain on the Inference Engine CPU backend: obs float[1, N] in, actions
    /// float[1, A] out, the observation normaliser baked into the graph (every trainer's
    /// export does that).
    ///
    /// Never throws for a bad or missing brain. It records why in <see cref="Error"/> and
    /// stays not-ready; the pilot then holds the racer at its rest pose, which reads in the
    /// HUD and the report as a brain problem rather than as a crash mid-race.
    /// </summary>
    public sealed class CreaturePolicy : IDisposable
    {
        private const string OUTPUT_NAME = "actions";

        private readonly Worker _worker;
        private readonly Tensor<float> _input;
        private readonly int _outputIndex;
        private readonly int _actionSize;
        private bool _faulted;
        private bool _disposed;

        private CreaturePolicy(Worker worker, Tensor<float> input, int outputIndex, int actionSize)
        {
            _worker = worker;
            _input = input;
            _outputIndex = outputIndex;
            _actionSize = actionSize;
            Error = string.Empty;
        }

        private CreaturePolicy(string error)
        {
            Error = error;
        }

        public bool IsReady => _worker != null && !_faulted && !_disposed;
        public string Error { get; private set; }

        public static CreaturePolicy Create(ModelAsset asset, string label, string expectedAssetPath,
                                            int observationSize, int actionSize)
        {
            if (asset == null)
            {
                return new CreaturePolicy(
                    $"{label}: no brain assigned. Copy the exported ONNX to {expectedAssetPath} and assign it as "
                    + "this racer's Brain in the race settings asset.");
            }

            Model model;
            try
            {
                model = ModelLoader.Load(asset);
            }
            catch (Exception exception)
            {
                return new CreaturePolicy($"{label}: {asset.name} did not load: {exception.Message}");
            }

            if (model.inputs == null || model.inputs.Count != 1)
            {
                return new CreaturePolicy($"{label}: {asset.name} has {model.inputs?.Count ?? 0} inputs, expected 1 ('obs')");
            }
            int inputWidth;
            try
            {
                DynamicTensorShape inputShape = model.inputs[0].shape;
                inputWidth = inputShape.rank >= 1 ? inputShape.Get(-1) : 0;
            }
            catch (Exception exception)
            {
                return new CreaturePolicy($"{label}: {asset.name} input shape is unreadable: {exception.Message}");
            }
            if (inputWidth != observationSize)
            {
                return new CreaturePolicy(
                    $"{label}: {asset.name} input '{model.inputs[0].name}' is {inputWidth} wide, "
                  + $"the creature's observation is {observationSize}");
            }

            int outputIndex = 0;
            for (int index = 0; index < model.outputs.Count; index++)
            {
                if (model.outputs[index].name == OUTPUT_NAME)
                {
                    outputIndex = index;
                    break;
                }
            }

            try
            {
                // CPU on purpose, like every other racer here: tiny MLPs at 50 Hz, and a GPU
                // round trip per step would cost more than the network.
                var worker = new Worker(model, BackendType.CPU);
                var input = new Tensor<float>(new TensorShape(1, observationSize));
                return new CreaturePolicy(worker, input, outputIndex, actionSize);
            }
            catch (Exception exception)
            {
                return new CreaturePolicy($"{label}: could not start a worker for {asset.name}: {exception.Message}");
            }
        }

        /// <summary>
        /// Runs one inference. Returns false, once and loudly, if the graph misbehaves; the
        /// policy then stays retired rather than logging every physics step.
        /// </summary>
        public bool Run(float[] obs, float[] actions)
        {
            if (!IsReady)
            {
                return false;
            }
            _input.Upload(obs);
            _worker.Schedule(_input);
            var output = _worker.PeekOutput(_outputIndex) as Tensor<float>;
            if (output == null)
            {
                Fault("the worker produced no float output");
                return false;
            }
            output.CompleteAllPendingOperations();
            TensorShape shape = output.shape;
            if (shape.length < _actionSize)
            {
                Fault($"the output has {shape.length} values, expected {_actionSize}");
                return false;
            }
            bool batched = shape.rank >= 2;
            for (int actionIndex = 0; actionIndex < _actionSize; actionIndex++)
            {
                actions[actionIndex] = batched ? output[0, actionIndex] : output[actionIndex];
            }
            return true;
        }

        public void Dispose()
        {
            _disposed = true;
            _worker?.Dispose();
            _input?.Dispose();
        }

        private void Fault(string reason)
        {
            _faulted = true;
            Error = reason;
            Debug.LogError($"[CreatureRace] policy retired: {reason}. The racer holds its rest pose from here.");
        }
    }
}
