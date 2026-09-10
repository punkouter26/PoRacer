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

        /// <summary>
        /// The transform that actually travels down the track — what the camera frames and
        /// what progress, standings and the trail views are measured from.
        ///
        /// For most creatures this is the articulation root's transform. Two exceptions
        /// make it worth naming separately from <see cref="Root"/>: IsaacH1's prefab root
        /// is an inert container with the articulation starting at its `pelvis` child, and
        /// Fido has no ArticulationBody at all — MuJoCo moves his torso and leaves the
        /// imported container standing at the start line. Keyed off the prefab root, either
        /// would race perfectly while the camera framed an empty grid slot and every
        /// standing read row 0.
        /// </summary>
        Transform Body { get; }

        int MaxStep { get; set; }

        /// <summary>
        /// The start gate. True from spawn until the 3-2-1 countdown ends, so the
        /// grid settles in place instead of racing away before the clock starts.
        ///
        /// It had to exist because the countdown was display-only: CountdownValue
        /// was written by the spawner and read by the HUD and nothing else, so for
        /// the whole 2.4 s every policy was driving. Progress is measured from the
        /// grid origin and the clock is zeroed at GO, so whatever a racer covered
        /// in that window was distance it never had to spend time on — and the
        /// faster the creature, the more of the race it was given for free.
        ///
        /// How a racer holds is its own business, because the policies are balance
        /// controllers as much as locomotion controllers and simply switching one
        /// off drops a humanoid on its face. See each implementation.
        ///
        /// Never set during training: the training areas leave it false and the
        /// policy drives from the first frame, exactly as it always has.
        /// </summary>
        bool StartHeld { get; set; }

        /// <summary>
        /// The prefab's authored root orientation — the pose whose joint chain
        /// lies the way the rig was designed. The Centipede is authored
        /// lying along its body axis (90 deg on X); dropping that rotation
        /// stands its capsule chain up as a vertical tower that collapses into
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
