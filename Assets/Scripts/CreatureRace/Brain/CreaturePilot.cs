using System;
using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// Everything that happens between a racer's physics state and its joint targets, and the
    /// race telemetry derived from the same step. Plain C#; one per racer, any creature.
    ///
    /// A racer's view calls <see cref="Step"/> once per physics step - from MuJoCo's control
    /// callback or from PhysX's FixedUpdate - so every racer runs the same decimation, the
    /// same hold, the same clipping and the same race clock, counted in ITS OWN physics
    /// steps. That is what keeps finish times fair whichever simulator Unity runs first.
    ///
    /// Targets: rest + clip(action, -1, 1) * action scale, radians, action order. Held (before
    /// GO, after a failure, without a brain) means the rest pose. Nothing here ever stands a
    /// fallen racer up (AGENTS rule H): a fallen racer keeps its own policy and gets up by
    /// itself or lies there.
    /// </summary>
    public sealed class CreaturePilot
    {
        /// <summary>Time constant of the displayed speed's smoothing, seconds.</summary>
        private const float SPEED_TIME_CONSTANT = 0.5f;
        private const float MIN_CROSSING_STEP = 1e-6f;
        /// <summary>WORM_SPEC detail 12: the trainers' simulation-health guard, rad/s or m/s.</summary>
        private const float DIVERGENCE_SPEED = 500f;

        private readonly CreaturePolicy _policy;
        private readonly CreatureLayout _layout;
        private readonly string _label;
        private readonly bool _previousActionClipped;
        private readonly int _actionSize;
        private readonly float[] _obs;
        private readonly float[] _rawAction;
        private readonly float[] _clippedAction;
        private readonly float[] _previousAction;
        private readonly float[] _lastJointPositions;
        private readonly float[] _actionOverride;

        private bool _hasOverride;
        private bool _held = true;
        private bool _holdTargetsWritten;
        private int _stepCounter;
        private int _racingSteps;
        private float _previousDistance;
        private float _dt;
        private Vector3 _laneOrigin;
        private Vector3 _laneForward = Vector3.forward;
        private float _finishDistance = float.PositiveInfinity;
        private float _timeLimit = float.PositiveInfinity;

        public CreaturePilot(CreaturePolicy policy, string label, bool previousActionClipped, CreatureLayout layout)
        {
            _policy = policy;
            _layout = layout;
            _label = label ?? string.Empty;
            _previousActionClipped = previousActionClipped;
            _actionSize = layout.ActionSize;
            _obs = new float[layout.ObservationSize];
            _rawAction = new float[_actionSize];
            _clippedAction = new float[_actionSize];
            _previousAction = new float[_actionSize];
            _lastJointPositions = new float[_actionSize];
            _actionOverride = new float[_actionSize];
            _dt = layout.Rig.PhysicsDt;
            FailReason = string.Empty;
            ProbeBodyA = layout.LeadBody;
            ProbeBodyB = layout.LeadBody;
            layout.CopyRestPose(_lastJointPositions);
        }

        public CreatureRacerStatus Status { get; private set; }
        public string FailReason { get; private set; }
        /// <summary>True once the racer's simulator has called <see cref="Step"/> at least once.</summary>
        public bool PhysicsReady { get; private set; }
        public float Distance { get; private set; }
        /// <summary>Track length once finished, the distance at the whistle once timed out.</summary>
        public float ResultDistance { get; private set; }
        public float Speed { get; private set; }
        public float ElapsedSeconds { get; private set; }
        public float FinishTimeSeconds { get; private set; } = -1f;
        public Vector3 Lead { get; private set; }
        public Vector3 LeadAtRelease { get; private set; }
        public float ReferenceHeight { get; private set; }
        public float ReferenceUpright { get; private set; } = 1f;
        /// <summary>Lowest reference-body uprightness since GO (below 0.5: it fell over).</summary>
        public float MinUprightSinceRelease { get; private set; } = 1f;
        public Vector3 ProbeOffset { get; private set; }
        public Vector3 ProbeOffsetAtRelease { get; private set; }
        /// <summary>Largest |q - rest| of any joint since GO, radians.</summary>
        public float MaxAbsJointSinceRelease { get; private set; }
        public int ProbeBodyA { get; private set; }
        public int ProbeBodyB { get; private set; }
        /// <summary>Point on <see cref="ProbeBodyB"/>, MuJoCo body frame.</summary>
        public Vector3 ProbePoint { get; private set; }

        public bool IsDone => Status == CreatureRacerStatus.Finished
                           || Status == CreatureRacerStatus.TimedOut
                           || Status == CreatureRacerStatus.Failed;

        public float LastJointPosition(int actionIndex)
        {
            return _lastJointPositions[actionIndex];
        }

        public void Configure(Vector3 laneOrigin, Vector3 laneForward, float finishDistance,
                              float timeLimit, float physicsDt)
        {
            _laneOrigin = laneOrigin;
            _laneForward = laneForward.sqrMagnitude > 0f ? laneForward.normalized : Vector3.forward;
            _finishDistance = finishDistance;
            _timeLimit = timeLimit;
            _dt = physicsDt > 0f ? physicsDt : _layout.Rig.PhysicsDt;
        }

        /// <summary>Which offset the view measures for the self-tests (body indices of the rig).</summary>
        public void SetProbe(int bodyA, int bodyB, Vector3 pointOnB)
        {
            ProbeBodyA = bodyA;
            ProbeBodyB = bodyB;
            ProbePoint = pointOnB;
        }

        /// <summary>Back to the grid: held at rest, no history, no telemetry.</summary>
        public void ResetForRace()
        {
            _held = true;
            _holdTargetsWritten = false;
            _hasOverride = false;
            _stepCounter = 0;
            _racingSteps = 0;
            _previousDistance = 0f;
            Array.Clear(_previousAction, 0, _previousAction.Length);
            Array.Clear(_rawAction, 0, _rawAction.Length);
            Array.Clear(_clippedAction, 0, _clippedAction.Length);
            _layout.CopyRestPose(_lastJointPositions);
            Status = CreatureRacerStatus.Waiting;
            FailReason = string.Empty;
            PhysicsReady = false;
            Distance = 0f;
            ResultDistance = 0f;
            Speed = 0f;
            ElapsedSeconds = 0f;
            FinishTimeSeconds = -1f;
            MaxAbsJointSinceRelease = 0f;
            MinUprightSinceRelease = 1f;
            ReferenceUpright = 1f;
        }

        /// <summary>
        /// Replaces the policy with fixed actions (self-tests). Takes effect at release and
        /// lasts until the next <see cref="ResetForRace"/>.
        /// </summary>
        public void SetActionOverride(float[] actions)
        {
            Array.Copy(actions, _actionOverride, _actionSize);
            _hasOverride = true;
        }

        /// <summary>GO. The very next physics step runs the policy.</summary>
        public void Release()
        {
            if (Status == CreatureRacerStatus.Failed)
            {
                return;
            }
            _held = false;
            _holdTargetsWritten = false;
            _stepCounter = 0;
            _racingSteps = 0;
            _previousDistance = Distance;
            Array.Clear(_previousAction, 0, _previousAction.Length);
            LeadAtRelease = Lead;
            ProbeOffsetAtRelease = ProbeOffset;
            MaxAbsJointSinceRelease = 0f;
            MinUprightSinceRelease = ReferenceUpright;
            Status = CreatureRacerStatus.Racing;
        }

        /// <summary>
        /// Out of the race, for good. Never followed by a rescue (AGENTS rule H): a failed
        /// racer is held at rest where it lies until the race is torn down.
        /// </summary>
        public void Fail(string reason)
        {
            if (Status == CreatureRacerStatus.Failed)
            {
                return;
            }
            Status = CreatureRacerStatus.Failed;
            FailReason = reason;
            ResultDistance = Distance;
            _held = true;
            _holdTargetsWritten = false;
            Debug.LogError($"[CreatureRace] racer out ({_label}): {reason}");
        }

        /// <summary>
        /// One physics step. Fills <paramref name="targets"/> (radians, action order) and
        /// returns true when they changed, so a PhysX view only rewrites its drives on a
        /// policy step. A MuJoCo view writes ctrl every step regardless.
        /// </summary>
        public bool Step(in CreatureBodyState body, float[] jointPositions, float[] jointVelocities,
                         in CreatureProbe probe, float[] targets)
        {
            PhysicsReady = true;
            // Telemetry only ever takes finite values, so a diverged racer leaves its last
            // good position on the HUD, in the report and under the camera, not NaN.
            float distance = Vector3.Dot(probe.Lead - _laneOrigin, _laneForward);
            bool finiteProbe = float.IsFinite(distance) && float.IsFinite(probe.ReferenceHeight)
                            && float.IsFinite(probe.ReferenceUpright) && Finite(probe.ProbeOffset);
            if (finiteProbe)
            {
                Lead = probe.Lead;
                ReferenceHeight = probe.ReferenceHeight;
                ReferenceUpright = probe.ReferenceUpright;
                ProbeOffset = probe.ProbeOffset;
                Distance = distance;
            }
            if (AllFinite(jointPositions))
            {
                Array.Copy(jointPositions, _lastJointPositions, _actionSize);
            }

            if (Status == CreatureRacerStatus.Failed)
            {
                return WriteHold(targets);
            }
            if (!finiteProbe || !body.IsFinite() || !AllFinite(jointPositions) || !AllFinite(jointVelocities))
            {
                Fail("non-finite physics state (the simulation diverged)");
                return WriteHold(targets);
            }
            if (ExceedsHealthLimit(body, jointVelocities))
            {
                Fail($"diverged: a joint or body speed above {DIVERGENCE_SPEED} (WORM_SPEC detail 12)");
                return WriteHold(targets);
            }
            if (_held)
            {
                _previousDistance = Distance;
                return WriteHold(targets);
            }

            AdvanceRaceClock();

            bool changed = false;
            if (_stepCounter % _layout.Rig.Decimation == 0)
            {
                Decide(body, jointPositions, jointVelocities);
                float scale = _layout.ActionScale;
                for (int actionIndex = 0; actionIndex < _actionSize; actionIndex++)
                {
                    targets[actionIndex] = _layout.Rest(actionIndex) + _clippedAction[actionIndex] * scale;
                }
                changed = true;
            }
            _stepCounter++;

            for (int actionIndex = 0; actionIndex < _actionSize; actionIndex++)
            {
                float magnitude = Mathf.Abs(jointPositions[actionIndex] - _layout.Rest(actionIndex));
                if (magnitude > MaxAbsJointSinceRelease)
                {
                    MaxAbsJointSinceRelease = magnitude;
                }
            }
            if (ReferenceUpright < MinUprightSinceRelease)
            {
                MinUprightSinceRelease = ReferenceUpright;
            }
            return changed;
        }

        private void Decide(in CreatureBodyState body, float[] jointPositions, float[] jointVelocities)
        {
            CreatureObservation.Build(_layout, body, jointPositions, jointVelocities, _previousAction, _obs);

            if (_hasOverride)
            {
                Array.Copy(_actionOverride, _rawAction, _actionSize);
            }
            else if (_policy == null || !_policy.Run(_obs, _rawAction))
            {
                // No brain, or it just faulted: hold the rest pose rather than improvise.
                Array.Clear(_rawAction, 0, _rawAction.Length);
            }

            for (int actionIndex = 0; actionIndex < _actionSize; actionIndex++)
            {
                float raw = _rawAction[actionIndex];
                if (!float.IsFinite(raw))
                {
                    raw = 0f;
                }
                float clipped = Mathf.Clamp(raw, -1f, 1f);
                _clippedAction[actionIndex] = clipped;
                _previousAction[actionIndex] = _previousActionClipped ? clipped : raw;
            }
        }

        /// <summary>
        /// The state this step sees is <c>_racingSteps * dt</c> after GO. A crossing between
        /// the last observation and this one is interpolated to a sub-step finish time.
        /// </summary>
        private void AdvanceRaceClock()
        {
            float now = _racingSteps * _dt;
            ElapsedSeconds = now;

            if (_racingSteps > 0)
            {
                float travelled = Distance - _previousDistance;
                float instantSpeed = travelled / _dt;
                Speed += (instantSpeed - Speed) * Mathf.Clamp01(_dt / SPEED_TIME_CONSTANT);

                if (Status == CreatureRacerStatus.Racing && _previousDistance < _finishDistance
                    && Distance >= _finishDistance)
                {
                    float fraction = travelled > MIN_CROSSING_STEP
                        ? (_finishDistance - _previousDistance) / travelled
                        : 1f;
                    FinishTimeSeconds = now - _dt + Mathf.Clamp01(fraction) * _dt;
                    ResultDistance = _finishDistance;
                    Status = CreatureRacerStatus.Finished;
                }
            }

            if (Status == CreatureRacerStatus.Racing)
            {
                ResultDistance = Distance;
                if (now >= _timeLimit)
                {
                    Status = CreatureRacerStatus.TimedOut;
                }
            }

            _previousDistance = Distance;
            _racingSteps++;
        }

        private bool WriteHold(float[] targets)
        {
            if (_holdTargetsWritten)
            {
                return false;
            }
            _layout.CopyRestPose(targets);
            _holdTargetsWritten = true;
            return true;
        }

        private static bool ExceedsHealthLimit(in CreatureBodyState body, float[] jointVelocities)
        {
            if (body.LinearVelocity.magnitude > DIVERGENCE_SPEED
                || body.AngularVelocity.magnitude > DIVERGENCE_SPEED)
            {
                return true;
            }
            for (int index = 0; index < jointVelocities.Length; index++)
            {
                if (Mathf.Abs(jointVelocities[index]) > DIVERGENCE_SPEED)
                {
                    return true;
                }
            }
            return false;
        }

        private static bool AllFinite(float[] values)
        {
            for (int index = 0; index < values.Length; index++)
            {
                if (!float.IsFinite(values[index]))
                {
                    return false;
                }
            }
            return true;
        }

        private static bool Finite(Vector3 value)
        {
            return float.IsFinite(value.x) && float.IsFinite(value.y) && float.IsFinite(value.z);
        }
    }
}
