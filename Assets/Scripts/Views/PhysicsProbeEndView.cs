using System.Diagnostics;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Tail half of the fixed-loop probe: runs after every other script's
    /// FixedUpdate (execution order 32000) and hands the exit timestamp to
    /// <see cref="PhysicsProbeView"/>, which owns the arithmetic. Added and
    /// removed as a pair with it — on its own it measures nothing.
    /// </summary>
    [DefaultExecutionOrder(32000)]
    [DisallowMultipleComponent]
    internal sealed class PhysicsProbeEndView : MonoBehaviour
    {
        private void FixedUpdate()
        {
            PhysicsProbeView.ReportStepEnd(Stopwatch.GetTimestamp());
        }
    }
}
