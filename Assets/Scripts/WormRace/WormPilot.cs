using System;
using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// Everything that happens between a worm's physics state and its joint targets, and
    /// the race telemetry derived from the same step. Plain C#; one per racer.
    ///
    /// A worm view calls <see cref="Step"/> once per physics step - from MuJoCo's control
    /// callback or from PhysX's FixedUpdate - so both worms run the same decimation, the
    /// same hold, the same clipping and the same race clock, counted in THEIR OWN physics
    /// steps. That last point is what makes the finish times fair: neither worm's time
    /// depends on the order Unity happens to run the two simulators in within a frame.
    /// </summary>
    internal sealed class WormPilot
    {
        /// <summary>Time constant of the displayed speed's smoothing, seconds.</summary>
        private const float SPEED_TIME_CONSTANT = 0.5f;
        private const float MIN_CROSSING_STEP = 1e-6f;
        /// <summary>WORM_SPEC detail 12: the trainers' simulation-health guard, rad/s or m/s.</summary>
        private const float DIVERGENCE_SPEED = 500f;

        private readonly WormPolicy _policy;
        private readonly string _label;
        private readonly bool _previousActionClipped;
        private readonly float[] _obs = new float[WormContract.OBS_SIZE];
        private readonly float[] _rawAction = new float[WormContract.ACTION_SIZE];
        private readonly float[] _clippedAction = new float[WormContract.ACTION_SIZE];
        private readonly float[] _previousAction = new float[WormContract.ACTION_SIZE];
        private readonly float[] _lastJointPositions = new float[WormContract.ACTION_SIZE];
        private readonly float[] _actionOverride = new float[WormContract.ACTION_SIZE];

        private bool _hasOverride;
        private bool _held = true;
        private bool _holdTargetsWritten;
        private int _stepCounter;
        private int _racingSteps;
        private float _previousDistance;
        private float _dt = WormContract.PHYSICS_DT;
        private Vector3 _laneOrigin;
        private Vector3 _laneForward = Vector3.forward;
        private float _finishDistance = float.PositiveInfinity;
        private float _timeLimit = float.PositiveInfinity;

        public WormPilot(WormPolicy policy, string label, bool previousActionClipped)
        {
            _policy = policy;
            _label = label ?? string.Empty;
            _previousActionClipped = previousActionClipped;
            FailReason = string.Empty;
        }

        public WormRacerStatus Status { get; private set; }
        public string FailReason { get; private set; }
        /// <summary>True once the worm's simulator has called <see cref="Step"/> at least once.</summary>
        public bool PhysicsReady { get; private set; }
        public float Distance { get; private set; }
        /// <summary>Track length once finished, the distance at the whistle once timed out.</summary>
        public float ResultDistance { get; private set; }
        public float Speed { get; private set; }
        public float ElapsedSeconds { get; private set; }
        public float FinishTimeSeconds { get; private set; } = -1f;
        public Vector3 Nose { get; private set; }
        public Vector3 NoseAtRelease { get; private set; }
        public float ReferenceHeight { get; private set; }
        public float SecondSegmentLateral { get; private set; }
        public float SecondSegmentVertical { get; private set; }
        public float MaxAbsJointSinceRelease { get; private set; }

        public bool IsDone => Status == WormRacerStatus.Finished
                           || Status == WormRacerStatus.TimedOut
                           || Status == WormRacerStatus.Failed;

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
            _dt = physicsDt > 0f ? physicsDt : WormContract.PHYSICS_DT;
        }

        /// <summary>Back to the grid: held straight, no history, no telemetry.</summary>
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
            Array.Clear(_lastJointPositions, 0, _lastJointPositions.Length);
            Status = WormRacerStatus.Waiting;
            FailReason = string.Empty;
            PhysicsReady = false;
            Distance = 0f;
            ResultDistance = 0f;
            Speed = 0f;
            ElapsedSeconds = 0f;
            FinishTimeSeconds = -1f;
            MaxAbsJointSinceRelease = 0f;
        }

        /// <summary>
        /// Replaces the policy with fixed actions (self-tests). Takes effect at release and
        /// lasts until the next <see cref="ResetForRace"/>.
        /// </summary>
        public void SetActionOverride(float[] actions)
        {
            Array.Copy(actions, _actionOverride, WormContract.ACTION_SIZE);
            _hasOverride = true;
        }

        /// <summary>GO. The very next physics step runs the policy.</summary>
        public void Release()
        {
            if (Status == WormRacerStatus.Failed)
            {
                return;
            }
            _held = false;
            _holdTargetsWritten = false;
            _stepCounter = 0;
            _racingSteps = 0;
            _previousDistance = Distance;
            Array.Clear(_previousAction, 0, _previousAction.Length);
            NoseAtRelease = Nose;
            MaxAbsJointSinceRelease = 0f;
            Status = WormRacerStatus.Racing;
        }

        /// <summary>
        /// Out of the race, for good. Never followed by a rescue (AGENTS rule H): a failed
        /// worm is held straight where it lies until the race is torn down.
        /// </summary>
        public void Fail(string reason)
        {
            if (Status == WormRacerStatus.Failed)
            {
                return;
            }
            Status = WormRacerStatus.Failed;
            FailReason = reason;
            ResultDistance = Distance;
            _held = true;
            _holdTargetsWritten = false;
            Debug.LogError($"[WormRace] racer out ({_label}): {reason}");
        }

        /// <summary>
        /// One physics step. Fills <paramref name="targets"/> (radians, action order) and
        /// returns true when they changed, so a PhysX view only rewrites its drives on a
        /// policy step. A MuJoCo view writes ctrl every step regardless.
        /// </summary>
        public bool Step(in WormBodyState body, float[] jointPositions, float[] jointVelocities,
                         in WormProbe probe, float[] targets)
        {
            PhysicsReady = true;
            // Telemetry only ever takes finite values, so a diverged worm leaves its last
            // good position on the HUD, in the report and under the camera, not NaN.
            float distance = Vector3.Dot(probe.Nose - _laneOrigin, _laneForward);
            bool finiteProbe = float.IsFinite(distance) && float.IsFinite(probe.ReferenceHeight)
                            && float.IsFinite(probe.SecondSegmentLateral)
                            && float.IsFinite(probe.SecondSegmentVertical);
            if (finiteProbe)
            {
                Nose = probe.Nose;
                ReferenceHeight = probe.ReferenceHeight;
                SecondSegmentLateral = probe.SecondSegmentLateral;
                SecondSegmentVertical = probe.SecondSegmentVertical;
                Distance = distance;
            }
            if (AllFinite(jointPositions))
            {
                Array.Copy(jointPositions, _lastJointPositions, WormContract.ACTION_SIZE);
            }

            if (Status == WormRacerStatus.Failed)
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
            if (_stepCounter % WormContract.DECIMATION == 0)
            {
                Decide(body, jointPositions, jointVelocities);
                for (int actionIndex = 0; actionIndex < WormContract.ACTION_SIZE; actionIndex++)
                {
                    targets[actionIndex] = _clippedAction[actionIndex] * WormContract.JOINT_RANGE_RAD;
                }
                changed = true;
            }
            _stepCounter++;

            for (int actionIndex = 0; actionIndex < WormContract.ACTION_SIZE; actionIndex++)
            {
                float magnitude = Mathf.Abs(jointPositions[actionIndex]);
                if (magnitude > MaxAbsJointSinceRelease)
                {
                    MaxAbsJointSinceRelease = magnitude;
                }
            }
            return changed;
        }

        private void Decide(in WormBodyState body, float[] jointPositions, float[] jointVelocities)
        {
            WormObservation.Build(body, jointPositions, jointVelocities, _previousAction, _obs);

            if (_hasOverride)
            {
                Array.Copy(_actionOverride, _rawAction, WormContract.ACTION_SIZE);
            }
            else if (_policy == null || !_policy.Run(_obs, _rawAction))
            {
                // No brain, or it just faulted: hold straight rather than improvise.
                Array.Clear(_rawAction, 0, _rawAction.Length);
            }

            for (int actionIndex = 0; actionIndex < WormContract.ACTION_SIZE; actionIndex++)
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

                if (Status == WormRacerStatus.Racing && _previousDistance < _finishDistance
                    && Distance >= _finishDistance)
                {
                    float fraction = travelled > MIN_CROSSING_STEP
                        ? (_finishDistance - _previousDistance) / travelled
                        : 1f;
                    FinishTimeSeconds = now - _dt + Mathf.Clamp01(fraction) * _dt;
                    ResultDistance = _finishDistance;
                    Status = WormRacerStatus.Finished;
                }
            }

            if (Status == WormRacerStatus.Racing)
            {
                ResultDistance = Distance;
                if (now >= _timeLimit)
                {
                    Status = WormRacerStatus.TimedOut;
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
            Array.Clear(targets, 0, WormContract.ACTION_SIZE);
            _holdTargetsWritten = true;
            return true;
        }

        private static bool ExceedsHealthLimit(in WormBodyState body, float[] jointVelocities)
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
    }
}
