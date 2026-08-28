using UnityEngine;

namespace IsaacSpider
{
    /// <summary>
    /// Pluggable target source for <see cref="IsaacSpiderAgent"/>. The policy only needs a world
    /// position (obs[41..42] are the horizontal offset in the body's yaw frame), so any waypoint,
    /// racer or player object can drive the spider without re-exporting the policy.
    /// </summary>
    public interface ITargetProvider
    {
        /// <summary>Returns false when no target is available; the agent then falls back to the Isaac ring sampler.</summary>
        bool TryGetTargetPosition(out Vector3 worldPosition);
    }
}
