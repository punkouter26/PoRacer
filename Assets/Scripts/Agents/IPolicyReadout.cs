using System.Collections.Generic;

namespace PoRacer.Agents
{
    /// <summary>
    /// A racer whose policy outputs can be shown to the viewer: the telemetry card's
    /// "what the brain is doing" bars read these, one bar per actuated degree of freedom.
    ///
    /// Read-only and allocation-free by design — implementations hand back the array the
    /// policy already writes, not a copy — so the telemetry system can sample every racer
    /// every frame. The values are the brain's raw commands, nominally in [-1, 1]; an
    /// Isaac Lab export can overshoot that range, so display code clamps rather than
    /// trusting it.
    /// </summary>
    public interface IPolicyReadout
    {
        /// <summary>The latest policy outputs, or null before the first decision.</summary>
        IReadOnlyList<float> LastActions { get; }
    }
}
