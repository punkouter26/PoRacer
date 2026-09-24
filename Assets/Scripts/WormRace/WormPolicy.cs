// SUPERSEDED by Assets/Scripts/CreatureRace/Brain/CreaturePolicy.cs (creature template, 2026-09-24).
// Kept out of compilation, content unchanged, until someone deletes this file and its .meta.
#if PORACER_WORMRACE_LEGACY
using System;
using Unity.InferenceEngine;
using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// One exported worm brain on the Inference Engine CPU backend: obs float[1, 35] in,
    /// actions float[1, 8] out, the observation normaliser baked into the graph.
    ///
    /// Never throws for a bad or missing brain. It records why in <see cref="Error"/> and
    /// stays not-ready, and the pilot then holds the worm straight, which reads in the HUD
    /// and the report as a brain problem rather than as a crash mid-race.
    /// </summary>
    internal sealed class WormPolicy : IDisposable
    {
        private readonly Worker _worker;
        private readonly Tensor<float> _input;
        private readonly int _outputIndex;
        private bool _faulted;
        private bool _disposed;

        public bool IsReady => _worker != null && !_faulted && !_disposed;
        public string Error { get; private set; }

        private WormPolicy(Worker worker, Tensor<float> input, int outputIndex)
        {
            _worker = worker;
            _input = input;
            _outputIndex = outputIndex;
            Error = string.Empty;
        }

        private WormPolicy(string error)
        {
            Error = error;
        }

        public static WormPolicy Create(ModelAsset asset, string label, string expectedAssetPath)
        {
            if (asset == null)
            {
                return new WormPolicy(
                    $"{label}: no brain assigned. Copy the exported ONNX to {expectedAssetPath} "
                  + "and re-run PoRacer.WormRace.EditorTools.Editor_BuildWormRaceScene.Build().");
            }

            Model model;
            try
            {
                model = ModelLoader.Load(asset);
            }
            catch (Exception exception)
            {
                return new WormPolicy($"{label}: {asset.name} did not load: {exception.Message}");
            }

            if (model.inputs == null || model.inputs.Count != 1)
            {
                return new WormPolicy($"{label}: {asset.name} has {model.inputs?.Count ?? 0} inputs, expected 1 ('obs')");
            }
            int inputWidth;
            try
            {
                DynamicTensorShape inputShape = model.inputs[0].shape;
                inputWidth = inputShape.rank >= 1 ? inputShape.Get(-1) : 0;
            }
            catch (Exception exception)
            {
                return new WormPolicy($"{label}: {asset.name} input shape is unreadable: {exception.Message}");
            }
            if (inputWidth != WormContract.OBS_SIZE)
            {
                return new WormPolicy(
                    $"{label}: {asset.name} input '{model.inputs[0].name}' is {inputWidth} wide, "
                  + $"the worm contract is {WormContract.OBS_SIZE}");
            }

            int outputIndex = 0;
            for (int index = 0; index < model.outputs.Count; index++)
            {
                if (model.outputs[index].name == WormContract.OUTPUT_NAME)
                {
                    outputIndex = index;
                    break;
                }
            }

            try
            {
                // CPU on purpose, like every other racer here: two tiny MLPs at 50 Hz, and a
                // GPU round trip per step would cost more than the network.
                var worker = new Worker(model, BackendType.CPU);
                var input = new Tensor<float>(new TensorShape(1, WormContract.OBS_SIZE));
                return new WormPolicy(worker, input, outputIndex);
            }
            catch (Exception exception)
            {
                return new WormPolicy($"{label}: could not start a worker for {asset.name}: {exception.Message}");
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
            if (shape.length < WormContract.ACTION_SIZE)
            {
                Fault($"the output has {shape.length} values, expected {WormContract.ACTION_SIZE}");
                return false;
            }
            bool batched = shape.rank >= 2;
            for (int actionIndex = 0; actionIndex < WormContract.ACTION_SIZE; actionIndex++)
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
            Debug.LogError($"[WormRace] policy retired: {reason}. The worm holds straight from here.");
        }
    }
}
#endif
