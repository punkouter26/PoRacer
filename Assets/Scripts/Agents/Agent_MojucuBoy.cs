using System;
using Creature.MojucuBoy;
using Mujoco;
using UnityEngine;

namespace PoRacer.Agents
{
    /// <summary>
    /// Race adapter for MojucuBoy, the MuJoCo humanoid: exposes
    /// <see cref="MojucuBoyController"/> through <see cref="ICreatureAgent"/> so the
    /// spawner, RacerView and camera treat him like any other catalog racer. Lives on
    /// the imported MJCF root.
    ///
    /// He is the second racer PhysX does not simulate, and he inherits every
    /// consequence documented on <see cref="Agent_Fido"/>:
    ///
    ///   * No ArticulationBody, so <see cref="Root"/> is null. Systems_Spawn and
    ///     RacerView read a null Root as "track the prefab root instead", and
    ///     <see cref="Body"/> corrects that to the hips, which is the body MuJoCo
    ///     actually moves.
    ///   * He shares the scene's single MjScene with Fido, so he collides with the
    ///     ground plane and with other MuJoCo racers, but not with PhysX racers and
    ///     not with the track's trigger volumes -- mud, boost and gust all key off
    ///     `attachedArticulationBody` and skip him rather than throw.
    ///
    /// Two things differ from Fido, both improvements:
    ///
    ///   * <see cref="RestRotation"/> is identity. build_mjcf.py authors the rig
    ///     facing MuJoCo +Y, which org.mujoco maps onto Unity +Z -- the race
    ///     direction -- so unlike Fido's -90 deg he needs no yaw correction at all.
    ///   * <see cref="SetGoal"/> is NOT a no-op. His observation carries a heading
    ///     command, so he can be steered toward the finish line. Fido cannot.
    ///
    /// Known limit, and it is a real one: the policy was trained with commanded
    /// headings within +/-0.6 rad of the racer's own facing, so his usable steering
    /// envelope is about +/-34 degrees. He holds a lane down a straight track well;
    /// he cannot take a sharp turn. Widening that needs a retrain, not a code change.
    /// </summary>
    [RequireComponent(typeof(MojucuBoyController))]
    [DisallowMultipleComponent]
    public sealed class Agent_MojucuBoy : MonoBehaviour, ICreatureAgent, IMujocoCreature, IAuthoredAppearance
    {
        /// <summary>
        /// Identity, deliberately. The MJCF is authored with a 180 degree facing yaw so
        /// the rig points down Unity +Z once org.mujoco's (x,y,z)->(x,z,y) mapping is
        /// applied. Fido needs -90 deg here because his MJCF builds along +X.
        /// </summary>
        private static readonly Quaternion Rest = Quaternion.identity;

        /// <summary>
        /// Hips height the policy trained to stand at, above the MuJoCo ground plane
        /// Systems_MujocoWorld puts at y = 0. Read off the rig's own stance solve
        /// (gate1_check.py), not guessed.
        /// </summary>
        private const float TRAINED_HIPS_HEIGHT = 0.7722f;

        // The env terminates an episode below 0.45 m, so a racer under that has
        // genuinely gone down rather than dipped through a stride.
        private const float FALLEN_HEIGHT = 0.45f;
        private const float FALLEN_UPRIGHT_DOT = 0.30f;
        private const float FALLEN_GRACE_SECONDS = 1.5f;

        private MojucuBoyController _controller;
        private Transform _hips;
        private Transform _goal;
        private float _fallenFor;
        private bool _failed;

        public bool Failed => _failed;

        /// <summary>Always null: MuJoCo simulates him, so no ArticulationBody exists.</summary>
        public ArticulationBody Root => null;

        /// <summary>
        /// The hips, which carry the MJCF free joint and are therefore the one thing
        /// MuJoCo actually moves. The imported container this component sits on never
        /// leaves the start line.
        /// </summary>
        public Transform Body => _hips != null ? _hips : transform;

        public int MaxStep { get; set; }

        public Quaternion RestRotation => Rest;

        private void Awake()
        {
            _controller = GetComponent<MojucuBoyController>();
            _hips = FindDeep(transform, "hips");
            if (_hips == null)
            {
                Debug.LogError($"[{name}] no 'hips' body under the MJCF root; "
                             + "camera and standings would track the start line.", this);
            }
            SnapToTrainedStance();
        }

        /// <summary>
        /// Cancels the spawner's few-centimetre drop and puts the hips at exactly the
        /// height the policy trained from.
        ///
        /// That drop exists so a PhysX creature is never born intersecting the ground,
        /// which costs a ragdoll nothing. A trained policy has no such slack: the same
        /// correction on Fido is the difference between walking away and collapsing on
        /// the first stride. He is also immune to the problem the drop solves, because
        /// MuJoCo resolves his contacts rather than PhysX.
        ///
        /// Runs in Awake so it lands before MjScene compiles the model in Start --
        /// CreateScene reads Unity transforms, so the corrected pose is what MuJoCo gets.
        /// </summary>
        private void SnapToTrainedStance()
        {
            if (_hips == null)
            {
                return;
            }
            float hipsOffset = _hips.position.y - transform.position.y;
            Vector3 position = transform.position;
            position.y = TRAINED_HIPS_HEIGHT - hipsOffset;
            transform.position = position;
        }

        private void FixedUpdate()
        {
            if (_failed || _hips == null)
            {
                return;
            }

            // Re-aim every physics tick: the heading command is relative to where he
            // currently is, so a goal set once at spawn would go stale the moment he
            // moved. Cheap -- it is an atan2.
            if (_goal != null)
            {
                _controller.SetGoal(_goal.position);
            }

            bool down = _hips.position.y < FALLEN_HEIGHT
                || Vector3.Dot(_hips.up, Vector3.up) < FALLEN_UPRIGHT_DOT;
            _fallenFor = down ? _fallenFor + Time.fixedDeltaTime : 0f;
            if (_fallenFor >= FALLEN_GRACE_SECONDS)
            {
                _failed = true;
            }
        }

        /// <summary>
        /// Unlike Fido, he observes a heading command and can be steered. Stored rather
        /// than applied once, because the heading has to be recomputed as he advances.
        /// </summary>
        public void SetGoal(Transform goal) => _goal = goal;

        public void SetAreaResetCallback(Action areaReset)
        {
        }

        /// <summary>
        /// No-op: joint power is MuJoCo actuator gain, not an ArticulationDrive, so the
        /// fatigue system has nothing to re-baseline here.
        /// </summary>
        public void NotifyDrivesChanged()
        {
        }

        private static Transform FindDeep(Transform root, string wanted)
        {
            if (root.name == wanted)
            {
                return root;
            }
            for (int i = 0; i < root.childCount; i++)
            {
                Transform found = FindDeep(root.GetChild(i), wanted);
                if (found != null)
                {
                    return found;
                }
            }
            return null;
        }
    }
}
