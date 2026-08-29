using UnityEngine;

namespace PoRacer.IsaacPorts
{
    /// <summary>
    /// Where a ported creature is trying to go. Implement this to drive one from your own
    /// game logic (waypoints, a navmesh path, a player transform, ...).
    ///
    /// Every port agent prefers, in order:
    ///   1. an explicit <c>Transform target</c>,
    ///   2. an <see cref="ITargetProvider"/> component on the same GameObject,
    ///   3. its own ring sampler as a fallback.
    ///
    /// This was four byte-for-byte copies, one per rig namespace, until the ports were
    /// merged into a single assembly on 2026-08-29.
    /// </summary>
    public interface ITargetProvider
    {
        /// <summary>World-space point to head for. Return false to stand still.</summary>
        bool TryGetTarget(out Vector3 worldPosition);
    }
}
