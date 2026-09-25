using System;
using System.Collections.Generic;
using PoRacer.Agents;
using PoRacer.Models;
using UnityEngine;
using VContainer.Unity;

namespace PoRacer.Systems
{
    /// <summary>
    /// Measures every racer for the broadcast: ground speed, how upright it is and whether
    /// it is falling or getting up, joint effort and power, energy spent, and what its
    /// brain is outputting. Writes <see cref="RaceTelemetryModel"/>; the camera director
    /// cuts on it and the telemetry card shows it.
    ///
    /// It only reads. Nothing here touches a body, a drive or a policy, so watching a
    /// racer can never change how it races.
    ///
    /// Effort comes from two places. A PhysX racer's ArticulationBody drives report the
    /// force they applied, which is read here directly; a MuJoCo racer has no
    /// ArticulationBody, so its controller measures the same from mjData and reports it
    /// through <see cref="IEffortReadout"/>. Energy is integrated per physics step
    /// (FixedTick) because power at 200 Hz changes far faster than the frame rate; the
    /// body's pose and the brain's outputs are read per frame (Tick), which is as often
    /// as anyone can see them.
    /// </summary>
    public sealed class Systems_RacerTelemetry : IFixedTickable, ITickable, IDisposable
    {
        private const float HISTORY_INTERVAL_SECONDS = 0.25f;
        private const float SPEED_SMOOTHING = 3f;
        private const float POSE_RATE_SMOOTHING = 8f;
        private const float POWER_SMOOTHING = 5f;
        // Weight of each new decision in the jitter average: ~7 decisions of memory.
        private const float JITTER_BLEND = 0.15f;
        // Below this the racer is on the ground; it has to pass the higher bar to count
        // as back up, so a racer teetering on the line does not flicker between states.
        private const float DOWN_UPRIGHT = 0.35f;
        private const float UP_UPRIGHT = 0.6f;
        private const float GETTING_UP_RISE_RATE = 0.25f;
        // Down for less than this and it was a stumble, not a recovery worth a caption.
        private const float MIN_RECOVERY_DOWN_SECONDS = 1.5f;
        // A drive limit at or above this is PhysX's "unlimited"; effort against it is meaningless.
        private const float UNLIMITED_FORCE = 1e6f;
        private const int MAX_DOFS_PER_JOINT = 3;

        private sealed class Tracked
        {
            public RacerTelemetry Telemetry;
            public Transform Body;
            public Vector3 RestUpLocal;
            public Vector3 LastPosition;
            public bool HasLastPosition;
            public float LastUpright;
            public ArticulationBody[] Joints;
            // MAX_DOFS_PER_JOINT slots per joint, in the drive's reduced-space order; 0 = no limit.
            public float[] ForceLimits;
            public IEffortReadout Effort;
            public IPolicyReadout Policy;
            public readonly float[] PreviousActions = new float[RacerTelemetry.MAX_ACTIONS];
        }

        private readonly RaceModel _raceModel;
        private readonly RaceTelemetryModel _model;
        private readonly List<Tracked> _tracked = new();
        private float _historyClock;

        public Systems_RacerTelemetry(RaceModel raceModel, RaceTelemetryModel model)
        {
            _raceModel = raceModel;
            _model = model;
        }

        /// <summary>
        /// Starts measuring a racer. Called by the spawner once the racer exists, while it
        /// still stands in its spawn pose — that pose defines "upright" for this body.
        /// </summary>
        public void Register(string racerId, ICreatureAgent agent, TrainingSource trainedBy)
        {
            if (agent == null || agent.Body == null)
            {
                return;
            }
            RacerTelemetry telemetry = _model.Add(racerId);
            telemetry.TrainedBy = trainedBy;
            telemetry.Upright = 1f;

            var tracked = new Tracked
            {
                Telemetry = telemetry,
                Body = agent.Body,
                // The body's own axis that points up in the spawn pose. Most rigs spawn with
                // their up along local +Y, but not all, and a fixed axis would read a rig
                // authored lying down as permanently fallen.
                RestUpLocal = Quaternion.Inverse(agent.Body.rotation) * Vector3.up,
                LastUpright = 1f,
                Effort = agent as IEffortReadout,
                Policy = agent as IPolicyReadout,
                Joints = Array.Empty<ArticulationBody>(),
                ForceLimits = Array.Empty<float>()
            };
            if (tracked.Effort == null && agent.Root != null)
            {
                CacheArticulation(tracked, agent.Root);
            }
            _tracked.Add(tracked);
        }

        /// <summary>Forgets the whole grid; the spawner calls this when it despawns.</summary>
        public void Clear()
        {
            _tracked.Clear();
            _model.Clear();
            _historyClock = 0f;
        }

        public void FixedTick()
        {
            float deltaTime = Time.fixedDeltaTime;
            float blend = 1f - Mathf.Exp(-POWER_SMOOTHING * deltaTime);
            for (int trackedIndex = 0; trackedIndex < _tracked.Count; trackedIndex++)
            {
                Tracked tracked = _tracked[trackedIndex];
                RacerTelemetry telemetry = tracked.Telemetry;
                float power;
                float effort;
                if (tracked.Effort != null)
                {
                    if (!tracked.Effort.HasEffort)
                    {
                        continue;
                    }
                    power = tracked.Effort.MechanicalPowerWatts;
                    effort = tracked.Effort.EffortFraction;
                    if (telemetry.MassKg <= 0f)
                    {
                        telemetry.MassKg = tracked.Effort.TotalMassKg;
                    }
                }
                else if (tracked.Joints.Length > 0)
                {
                    MeasureArticulation(tracked, out power, out effort);
                }
                else
                {
                    continue;
                }

                telemetry.HasPower = true;
                telemetry.PowerWatts = Mathf.Lerp(telemetry.PowerWatts, power, blend);
                telemetry.EffortFraction = effort < 0f
                    ? RacerTelemetry.UNKNOWN
                    : Mathf.Lerp(Mathf.Max(0f, telemetry.EffortFraction), effort, blend);

                // Energy only counts while the clock runs and the racer is still in it,
                // so cost of transport is work spent racing, not holding on the grid.
                if (_raceModel.RaceActive)
                {
                    RacerState racer = _raceModel.FindRacer(telemetry.RacerId);
                    if (racer != null && racer.Status == RacerStatus.Racing)
                    {
                        telemetry.EnergyJoules += power * deltaTime;
                    }
                }
            }
        }

        public void Tick()
        {
            float deltaTime = Time.deltaTime;
            if (deltaTime <= 0f)
            {
                return;
            }
            float speedBlend = 1f - Mathf.Exp(-SPEED_SMOOTHING * deltaTime);
            float rateBlend = 1f - Mathf.Exp(-POSE_RATE_SMOOTHING * deltaTime);
            float now = Time.unscaledTime;

            for (int trackedIndex = 0; trackedIndex < _tracked.Count; trackedIndex++)
            {
                Tracked tracked = _tracked[trackedIndex];
                if (tracked.Body == null)
                {
                    continue;
                }
                RacerTelemetry telemetry = tracked.Telemetry;

                Vector3 position = tracked.Body.position;
                if (tracked.HasLastPosition)
                {
                    Vector3 travel = position - tracked.LastPosition;
                    travel.y = 0f;
                    float speed = travel.magnitude / deltaTime;
                    if (float.IsFinite(speed))
                    {
                        telemetry.SpeedMps = Mathf.Lerp(telemetry.SpeedMps, speed, speedBlend);
                    }
                }
                tracked.LastPosition = position;
                tracked.HasLastPosition = true;

                float upright = Vector3.Dot(tracked.Body.rotation * tracked.RestUpLocal, Vector3.up);
                float rate = (upright - tracked.LastUpright) / deltaTime;
                tracked.LastUpright = upright;
                telemetry.Upright = upright;
                telemetry.FallRate = Mathf.Lerp(telemetry.FallRate, Mathf.Max(0f, -rate), rateBlend);
                telemetry.RiseRate = Mathf.Lerp(telemetry.RiseRate, Mathf.Max(0f, rate), rateBlend);
                UpdateDownState(telemetry, upright, deltaTime, now);

                RacerState racer = _raceModel.FindRacer(telemetry.RacerId);
                if (racer != null)
                {
                    telemetry.DistanceMeters = Mathf.Max(0f, racer.Progress);
                }

                ReadActions(tracked);
            }

            _historyClock += Time.unscaledDeltaTime;
            if (_historyClock >= HISTORY_INTERVAL_SECONDS)
            {
                _historyClock -= HISTORY_INTERVAL_SECONDS;
                for (int trackedIndex = 0; trackedIndex < _tracked.Count; trackedIndex++)
                {
                    RacerTelemetry telemetry = _tracked[trackedIndex].Telemetry;
                    telemetry.PushHistory(telemetry.SpeedMps, Mathf.Clamp01(telemetry.Upright),
                        Mathf.Max(0f, telemetry.EffortFraction));
                }
            }
        }

        public void Dispose()
        {
        }

        private static void UpdateDownState(RacerTelemetry telemetry, float upright, float deltaTime, float now)
        {
            if (!telemetry.IsDown)
            {
                if (upright < DOWN_UPRIGHT)
                {
                    telemetry.IsDown = true;
                    telemetry.DownSeconds = 0f;
                }
                telemetry.IsGettingUp = false;
                return;
            }
            telemetry.DownSeconds += deltaTime;
            if (upright > UP_UPRIGHT)
            {
                // Back up under its own policy — rule H: nobody stands a racer up.
                if (telemetry.DownSeconds >= MIN_RECOVERY_DOWN_SECONDS)
                {
                    telemetry.RecoveredAt = now;
                }
                telemetry.IsDown = false;
                telemetry.IsGettingUp = false;
                return;
            }
            telemetry.IsGettingUp = telemetry.RiseRate > GETTING_UP_RISE_RATE;
        }

        /// <summary>
        /// Copies the brain's outputs and scores how much they moved since the last
        /// decision. The copy only changes when the policy decides, so a frame with no new
        /// decision leaves the jitter average alone.
        /// </summary>
        private static void ReadActions(Tracked tracked)
        {
            IPolicyReadout policy = tracked.Policy;
            // The readout is an agent MonoBehaviour: Unity's == catches a destroyed one,
            // which ?. on the interface would not.
            IReadOnlyList<float> actions = policy is UnityEngine.Object unityObject && unityObject == null
                ? null
                : policy?.LastActions;
            RacerTelemetry telemetry = tracked.Telemetry;
            if (actions == null)
            {
                telemetry.ActionCount = 0;
                return;
            }
            int count = Mathf.Min(actions.Count, RacerTelemetry.MAX_ACTIONS);
            float change = 0f;
            for (int actionIndex = 0; actionIndex < count; actionIndex++)
            {
                float value = actions[actionIndex];
                value = float.IsFinite(value) ? Mathf.Clamp(value, -1f, 1f) : 0f;
                change += Mathf.Abs(value - tracked.PreviousActions[actionIndex]);
                tracked.PreviousActions[actionIndex] = value;
                telemetry.Actions[actionIndex] = value;
            }
            bool hadActions = telemetry.ActionCount == count;
            telemetry.ActionCount = count;
            if (hadActions && count > 0 && change > 0f)
            {
                telemetry.Jitter = Mathf.Lerp(telemetry.Jitter, change / count, JITTER_BLEND);
            }
        }

        private static void CacheArticulation(Tracked tracked, ArticulationBody root)
        {
            ArticulationBody[] bodies = root.GetComponentsInChildren<ArticulationBody>();
            float mass = 0f;
            int jointCount = 0;
            for (int bodyIndex = 0; bodyIndex < bodies.Length; bodyIndex++)
            {
                mass += bodies[bodyIndex].mass;
                if (!bodies[bodyIndex].isRoot && bodies[bodyIndex].dofCount > 0)
                {
                    jointCount++;
                }
            }
            tracked.Telemetry.MassKg = mass;
            tracked.Joints = new ArticulationBody[jointCount];
            tracked.ForceLimits = new float[jointCount * MAX_DOFS_PER_JOINT];
            int jointIndex = 0;
            for (int bodyIndex = 0; bodyIndex < bodies.Length; bodyIndex++)
            {
                ArticulationBody body = bodies[bodyIndex];
                if (body.isRoot || body.dofCount <= 0)
                {
                    continue;
                }
                tracked.Joints[jointIndex] = body;
                int slot = jointIndex * MAX_DOFS_PER_JOINT;
                if (body.jointType == ArticulationJointType.SphericalJoint)
                {
                    // Reduced-space order is the unlocked axes in X (twist), Y, Z order.
                    int dof = 0;
                    if (body.twistLock != ArticulationDofLock.LockedMotion)
                    {
                        tracked.ForceLimits[slot + dof++] = LimitOf(body.xDrive);
                    }
                    if (body.swingYLock != ArticulationDofLock.LockedMotion)
                    {
                        tracked.ForceLimits[slot + dof++] = LimitOf(body.yDrive);
                    }
                    if (body.swingZLock != ArticulationDofLock.LockedMotion && dof < MAX_DOFS_PER_JOINT)
                    {
                        tracked.ForceLimits[slot + dof] = LimitOf(body.zDrive);
                    }
                }
                else
                {
                    // Revolute and prismatic joints drive their single axis through xDrive.
                    tracked.ForceLimits[slot] = LimitOf(body.xDrive);
                }
                jointIndex++;
            }
        }

        /// <summary>
        /// The limit captured at spawn is the drive's full strength. Fatigue lowers the live
        /// limit during a race, so effort against the spawn value reads as "share of full
        /// strength", which is what a viewer means by effort.
        /// </summary>
        private static float LimitOf(ArticulationDrive drive)
        {
            return drive.forceLimit > 0f && drive.forceLimit < UNLIMITED_FORCE ? drive.forceLimit : 0f;
        }

        private static void MeasureArticulation(Tracked tracked, out float power, out float effort)
        {
            power = 0f;
            float effortSum = 0f;
            int limited = 0;
            for (int jointIndex = 0; jointIndex < tracked.Joints.Length; jointIndex++)
            {
                ArticulationBody joint = tracked.Joints[jointIndex];
                if (joint == null)
                {
                    continue;
                }
                ArticulationReducedSpace force = joint.driveForce;
                ArticulationReducedSpace velocity = joint.jointVelocity;
                int dofs = Mathf.Min(force.dofCount, MAX_DOFS_PER_JOINT);
                for (int dof = 0; dof < dofs; dof++)
                {
                    float jointForce = force[dof];
                    power += Mathf.Abs(jointForce * velocity[dof]);
                    float limit = tracked.ForceLimits[jointIndex * MAX_DOFS_PER_JOINT + dof];
                    if (limit > 0f)
                    {
                        effortSum += Mathf.Min(1f, Mathf.Abs(jointForce) / limit);
                        limited++;
                    }
                }
            }
            if (!float.IsFinite(power))
            {
                power = 0f;
            }
            effort = limited > 0 && float.IsFinite(effortSum) ? effortSum / limited : RacerTelemetry.UNKNOWN;
        }
    }
}
