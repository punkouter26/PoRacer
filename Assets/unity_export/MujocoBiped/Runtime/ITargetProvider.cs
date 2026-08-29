using UnityEngine;

namespace MujocoBiped
{
    /// <summary>
    /// Where the creature is trying to go. Implement this to drive it from your own game
    /// logic - a waypoint list, a race finish line, a player transform.
    /// <see cref="MujocoBipedAgent"/> prefers, in order:
    ///   1. an explicit <c>Transform target</c>,
    ///   2. an <see cref="ITargetProvider"/> component on the same GameObject,
    ///   3. <see cref="MujocoBipedTargetSampler"/> as a fallback.
    /// </summary>
    public interface ITargetProvider
    {
        /// <summary>World-space point to head for. Return false to stand still.</summary>
        bool TryGetTarget(out Vector3 worldPosition);
    }
}
