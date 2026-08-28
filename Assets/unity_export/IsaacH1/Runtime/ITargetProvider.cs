using UnityEngine;

namespace IsaacH1
{
    /// <summary>
    /// Where the creature is trying to go. Implement this to drive it from your own
    /// game logic (waypoints, a navmesh path, a player transform, ...).
    /// <see cref="IsaacH1Agent"/> prefers, in order:
    ///   1. an explicit <c>Transform target</c>,
    ///   2. an <see cref="ITargetProvider"/> component,
    ///   3. <see cref="IsaacH1RingTargetSampler"/> as a fallback.
    /// </summary>
    public interface ITargetProvider
    {
        /// <summary>World-space point to head for. Return false to stand still.</summary>
        bool TryGetTarget(out Vector3 worldPosition);
    }
}
