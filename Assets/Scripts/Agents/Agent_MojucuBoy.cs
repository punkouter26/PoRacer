using System;
using System.Collections.Generic;
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
    /// He is the racer PhysX does not simulate, and carries every consequence
    /// of that (these were first documented on Fido, removed 2026-09-10):
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
    /// Steering: on a lane he aims LANE_LOOKAHEAD metres ahead on the line from his
    /// start to his goal, and MojucuBoyController caps each correction at 30 degrees
    /// from where he faces. The lane-follow brains are trained on exactly that loop
    /// (mojucuboy_env.py --lane-follow); the older brains saw only a fixed heading.
    /// </summary>
    [RequireComponent(typeof(MojucuBoyController))]
    [DisallowMultipleComponent]
    public sealed class Agent_MojucuBoy : MonoBehaviour, ICreatureAgent, IMujocoCreature, IAuthoredAppearance,
        IPolicyReadout, IEffortReadout
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

        // Surface probe for the stance snap: start a little above him and look a short way
        // down, so the hit is the ground he was placed on rather than a lower switchback.
        private const float SURFACE_PROBE_UP = 1f;
        private const float SURFACE_PROBE_DISTANCE = 6f;

        /// <summary>
        /// Lane steering: metres ahead on his lane that the heading command aims at.
        /// MUST equal LANE_LOOKAHEAD in training/mojucuboy/mojucuboy_env.py, which trains
        /// him on exactly this pursuit; the correction cap lives in MojucuBoyController.
        /// </summary>
        private const float LANE_LOOKAHEAD = 3f;
        // A goal that has moved further than this since SetGoal is a course carrot, not
        // a fixed lane point, and is aimed at directly.
        private const float STATIC_GOAL_TOLERANCE = 0.05f;

        // Joystick brains: the lane turned into a velocity command each tick.
        private const float RACE_SPEED = 1.5f;       // m/s forward, the speed he races at
        private const float YAW_GAIN = 1.5f;         // rad/s of turn per radian off the aim point
        private const float SIDESTEP_GAIN = 0.5f;    // m/s of sidestep per metre the aim point is sideways
        private readonly RaycastHit[] _probeHits = new RaycastHit[8];


        private MojucuBoyController _controller;
        private Transform _hips;
        private Transform _goal;
        // The lane: from where he stood when the goal was set to where the goal was.
        private Vector3 _laneStart;
        private Vector3 _laneEnd;
        private bool _failed;
        private bool _startHeld;

        public bool Failed => _failed;

        /// <summary>
        /// Held by freezing the MuJoCo step, not by touching him: see
        /// <see cref="Systems.Systems_MujocoWorld.HoldStepping"/>. Nothing Unity-side
        /// can pin him — he has no ArticulationBody and MuJoCo owns his transforms —
        /// and parking the policy alone would drop a humanoid on the line. With the
        /// world not stepping he is frozen in his trained stance, exactly as spawned,
        /// which is a cleaner hold than any of the PhysX racers get.
        ///
        /// The spawner holds the world once for the whole grid, so this only records
        /// the flag and keeps the heading command from going stale against a position
        /// that is not changing.
        /// </summary>
        public bool StartHeld
        {
            get => _startHeld;
            set => _startHeld = value;
        }

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

        public IReadOnlyList<float> LastActions => _controller != null ? _controller.LastAction : null;

        public bool HasEffort => _controller != null && _controller.HasTelemetry;

        public float MechanicalPowerWatts => _controller != null ? _controller.MechanicalPowerWatts : 0f;

        public float EffortFraction => _controller != null ? _controller.EffortFraction : 0f;

        public float TotalMassKg => _controller != null ? _controller.TotalMassKg : 0f;

        private void Awake()
        {
            _controller = GetComponent<MojucuBoyController>();
            _hips = FindDeep(transform, "hips");
            if (_hips == null)
            {
                Debug.LogError($"[{name}] no 'hips' body under the MJCF root; "
                             + "camera and standings would track the start line.", this);
            }
            // MuJoCo physics but an ONNX policy: the observation normaliser lives
            // inside the graph, so he runs through the Inference Engine like the
            // Isaac ports rather than off a JSON MLP like Fido. 75 in, 21 out,
            // per MojucuBoyObservation and the rig actuator order.
            Agent_NativeBrainProbe.Attach(gameObject, "MojucuBoy", "ONNX/IE", 75, 21, _controller != null);
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
            // Relative to the surface UNDER him, not to absolute zero.
            //
            // This used to be `position.y = TRAINED_HIPS_HEIGHT - hipsOffset`, which is
            // only correct where the ground is at y = 0 — true of every builder map, and
            // wrong by 14.3 m on the Acrobat course, whose first centreline knot is that
            // far up. It teleported him off the road before he took a step: measured at
            // y = 0.8 while slab 0 of his own MuJoCo road sat at y = 13.9.
            //
            // A downward probe finds whichever surface the spawner placed him on, so the
            // trained stance height is preserved on flat ground and on a mountain road
            // alike. No hit (spawned over a void) keeps the old absolute behaviour, which
            // is the best guess available.
            position.y = TryFindSurfaceY(position, out float surfaceY)
                ? surfaceY + TRAINED_HIPS_HEIGHT - hipsOffset
                : TRAINED_HIPS_HEIGHT - hipsOffset;
            transform.position = position;
        }

        /// <summary>
        /// Height of the nearest surface below <paramref name="from"/>, ignoring his own
        /// colliders. Unity-side geometry, deliberately: the spawner placed him against a
        /// Unity collider (RaceCourseView.TrySurfaceAt probes the course's own road), so
        /// that is the surface his stance has to match even though MuJoCo will be the thing
        /// simulating him against its mirrored copy of it.
        /// </summary>
        private bool TryFindSurfaceY(Vector3 from, out float surfaceY)
        {
            surfaceY = 0f;
            int count = Physics.RaycastNonAlloc(from + Vector3.up * SURFACE_PROBE_UP, Vector3.down,
                _probeHits, SURFACE_PROBE_DISTANCE, ~0, QueryTriggerInteraction.Ignore);
            bool found = false;
            for (int hitIndex = 0; hitIndex < count; hitIndex++)
            {
                if (_probeHits[hitIndex].collider.transform.IsChildOf(transform))
                {
                    continue;
                }
                if (!found || _probeHits[hitIndex].point.y > surfaceY)
                {
                    surfaceY = _probeHits[hitIndex].point.y;
                    found = true;
                }
            }
            return found;
        }

        private void FixedUpdate()
        {
            if (_failed || _hips == null || _startHeld)
            {
                return;
            }

            // Re-aim every physics tick: the heading command is relative to where he
            // currently is, so a goal set once at spawn would go stale the moment he
            // moved. Cheap -- it is an atan2.
            if (_goal != null)
            {
                Steer(AimPoint());
            }

            // Being on the floor is NOT a failure. He is trained to get back up and
            // is never picked up by a marshal, so reporting failure the moment he
            // goes down would end his race before the recovery he is trained for
            // could start. RacerView's knockdown referee owns that call: still on
            // his back and going nowhere after its window, and he is a DNF.
            //
            // Failed stays reserved for what ICreatureAgent means by it -- a
            // physics failure the racer cannot come back from.
            if (!float.IsFinite(_hips.position.x) || !float.IsFinite(_hips.position.y)
                || !float.IsFinite(_hips.position.z))
            {
                _failed = true;
            }
        }

        /// <summary>
        /// Unlike Fido, he observes a heading command and can be steered. Stored rather
        /// than applied once, because the heading has to be recomputed as he advances.
        /// </summary>
        public void SetGoal(Transform goal)
        {
            _goal = goal;
            if (goal == null)
            {
                return;
            }
            _laneStart = Body.position;
            _laneEnd = goal.position;
            // Apply it immediately rather than waiting for the next FixedUpdate: the
            // policy's first observation must already carry a sane heading, or he spends
            // his opening strides correcting a course error he was never trained for.
            if (_controller != null)
            {
                Steer(AimPoint());
            }
        }

        /// <summary>Out of the race: the goal is dropped and the controller holds still.</summary>
        public void HoldStill()
        {
            _goal = null;
            if (_controller != null)
            {
                _controller.Hold();
            }
        }

        /// <summary>
        /// Points him at <paramref name="aim"/>. A heading brain is given the direction; a
        /// joystick brain is given a body-frame velocity: turn toward the aim point,
        /// sidestep by how far it lies to his side, and walk at race speed, easing off
        /// while the turn is large.
        /// </summary>
        private void Steer(Vector3 aim)
        {
            if (!_controller.IsJoystick)
            {
                _controller.SetGoal(aim, Body.position);
                return;
            }
            Vector3 toAim = aim - Body.position;
            toAim.y = 0f;
            Vector3 facing = Body.forward;
            facing.y = 0f;
            if (toAim.sqrMagnitude < 0.0001f || facing.sqrMagnitude < 0.0001f)
            {
                _controller.SetJoystick(RACE_SPEED, 0f, 0f);
                return;
            }
            facing.Normalize();
            // Unity is left-handed: a positive angle about +Y turns him RIGHT, while the
            // command's yaw rate is positive to the LEFT (MuJoCo's +Z up).
            float error = Vector3.SignedAngle(facing, toAim, Vector3.up) * Mathf.Deg2Rad;
            Vector3 left = Vector3.Cross(facing, Vector3.up);
            float sideways = Vector3.Dot(toAim, left);
            _controller.SetJoystick(RACE_SPEED * Mathf.Clamp01(Mathf.Cos(error)),
                                    SIDESTEP_GAIN * sideways,
                                    -YAW_GAIN * error);
        }

        /// <summary>
        /// Where the heading command points. On a fixed lane goal: LANE_LOOKAHEAD metres
        /// ahead of him ON THE LANE, so a racer who has drifted off it is steered back
        /// onto the line rather than merely toward the far goal (pure pursuit, as he was
        /// trained). A moving goal - a course carrot, which already rides the road ahead
        /// of him - is aimed at directly.
        /// </summary>
        private Vector3 AimPoint()
        {
            Vector3 goal = _goal.position;
            if ((goal - _laneEnd).sqrMagnitude > STATIC_GOAL_TOLERANCE * STATIC_GOAL_TOLERANCE)
            {
                return goal;
            }
            Vector3 lane = _laneEnd - _laneStart;
            lane.y = 0f;
            if (lane.sqrMagnitude < 0.0001f)
            {
                return goal;
            }
            Vector3 direction = lane.normalized;
            Vector3 offset = Body.position - _laneStart;
            float along = offset.x * direction.x + offset.z * direction.z;
            return _laneStart + direction * (along + LANE_LOOKAHEAD);
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
