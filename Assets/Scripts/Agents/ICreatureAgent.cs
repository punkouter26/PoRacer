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
        /// <summary>
        /// The prefab's authored root orientation — the pose whose joint chain
        /// lies the way the rig was designed. Snake and Centipede are authored
        /// lying along their body axis (90 deg on X); dropping that rotation
        /// stands their capsule chain up as a vertical tower that collapses into
        /// itself and blows the solver to NaN. Anything that re-orients a
        /// creature (spawn, episode reset, marshal rescue) must apply its heading
        /// on top of this, never instead of it.
        /// </summary>
        Quaternion RestRotation { get; }
        void SetGoal(Transform goal);
        void SetAreaResetCallback(Action areaReset);
        /// <summary>
        /// Call after externally rewriting joint drives (spawn/training quirks) so
        /// the fatigue system re-captures its full-power baseline from the new values.
        /// </summary>
        void NotifyDrivesChanged();
    }
}
