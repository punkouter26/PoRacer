using System;
using UnityEngine;

namespace PoRacer.Agents
{
    /// <summary>
    /// Contract every racing creature agent fulfils, so spawn/track/reset code
    /// stays morphology-agnostic.
    /// </summary>
    public interface ICreatureAgent
    {
        bool Failed { get; }
        ArticulationBody Root { get; }
        int MaxStep { get; set; }
        void SetGoal(Transform goal);
        void SetAreaResetCallback(Action areaReset);
        /// <summary>
        /// Call after externally rewriting joint drives (spawn/training quirks) so
        /// the fatigue system re-captures its full-power baseline from the new values.
        /// </summary>
        void NotifyDrivesChanged();
    }
}
