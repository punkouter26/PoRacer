using System.Diagnostics;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Wall-clock probe for the fixed loop, the part of the frame the diagnostic
    /// overlay was blind to. With ~13 articulation bodies per racer a full grid
    /// spends far more of its frame in PhysX than in rendering, and no render
    /// counter shows that.
    ///
    /// Runs first in every FixedUpdate (execution order -32000); the matching
    /// <see cref="PhysicsProbeEndView"/> runs last. Between the two lies every
    /// other script's FixedUpdate, and between one step's end and the next step's
    /// entry lies PhysX itself. Both spans are timed here.
    ///
    /// The PhysX span is only attributable when a second step follows inside the
    /// same frame — after the last step of a frame the gap also contains rendering,
    /// so that sample is dropped rather than reported as a lie. Multi-step frames
    /// are exactly the ones that are already missing their budget, which is when
    /// the number matters.
    /// </summary>
    [DefaultExecutionOrder(-32000)]
    [DisallowMultipleComponent]
    internal sealed class PhysicsProbeView : MonoBehaviour
    {
        private static readonly double MsPerTick = 1000.0 / Stopwatch.Frequency;

        /// <summary>Timestamp of the current step's entry; read by the end probe.</summary>
        internal static long StepEntryTicks;
        /// <summary>Timestamp of the previous step's exit, or 0 before the first exit.</summary>
        internal static long LastStepExitTicks;

        // Accumulated over the steps of the frame in progress.
        private static double _scriptMsThisFrame;
        private static double _physxMsThisFrame;
        private static int _stepsThisFrame;
        private static int _physxSamplesThisFrame;

        /// <summary>Script FixedUpdate time for the last completed frame, ms.</summary>
        internal static float ScriptMsPerFrame { get; private set; }
        /// <summary>PhysX solve time for the last completed frame, ms; -1 when unmeasurable.</summary>
        internal static float PhysxMsPerFrame { get; private set; } = -1f;
        /// <summary>Fixed steps that ran during the last completed frame.</summary>
        internal static int StepsPerFrame { get; private set; }

        /// <summary>Adds the probe pair to <paramref name="host"/> if it is not already there.</summary>
        internal static void EnsureOn(GameObject host)
        {
            if (!host.TryGetComponent(out PhysicsProbeView _))
            {
                host.AddComponent<PhysicsProbeView>();
            }
            if (!host.TryGetComponent(out PhysicsProbeEndView _))
            {
                host.AddComponent<PhysicsProbeEndView>();
            }
        }

        /// <summary>Called by the end probe once the last script FixedUpdate has run.</summary>
        internal static void ReportStepEnd(long exitTicks)
        {
            _scriptMsThisFrame += (exitTicks - StepEntryTicks) * MsPerTick;
            LastStepExitTicks = exitTicks;
        }

        private void FixedUpdate()
        {
            long entryTicks = Stopwatch.GetTimestamp();
            // Second and later steps of one frame: everything since the previous
            // step's exit was PhysX, with no rendering mixed in.
            if (_stepsThisFrame > 0 && LastStepExitTicks > 0L)
            {
                _physxMsThisFrame += (entryTicks - LastStepExitTicks) * MsPerTick;
                _physxSamplesThisFrame++;
            }
            StepEntryTicks = entryTicks;
            _stepsThisFrame++;
        }

        /// <summary>
        /// Update runs after the frame's fixed steps, so this is where a frame's
        /// worth of samples is published and the accumulators reset.
        /// </summary>
        private void Update()
        {
            ScriptMsPerFrame = (float)_scriptMsThisFrame;
            StepsPerFrame = _stepsThisFrame;
            PhysxMsPerFrame = _physxSamplesThisFrame > 0 ? (float)_physxMsThisFrame : -1f;

            _scriptMsThisFrame = 0.0;
            _physxMsThisFrame = 0.0;
            _stepsThisFrame = 0;
            _physxSamplesThisFrame = 0;
        }
    }
}
