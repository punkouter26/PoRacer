using System;
using System.Text;
using System.Threading;
using Cysharp.Threading.Tasks;
using MessagePipe;
using UnityEngine;
using UnityEngine.SceneManagement;
using VContainer.Unity;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// Runs a creature race scene: a series of N races, or one self-test, start to finish.
    /// The same flow for every creature (the worm, the quad, the next one); the creature
    /// comes in through its <see cref="ICreatureSpawner"/> and its <see cref="CreatureRaceConfig"/>.
    ///
    /// One racer per entry of the config's racer list (lane = index), each with its own pilot
    /// and brain; nothing here knows which lane runs in which simulator.
    ///
    /// Per race: spawn every racer at rest on the start line, a 3-2-1 countdown with every
    /// pilot held (physics runs, so all settle the same way; the policies do not), GO, then
    /// wait until each racer has finished, run out of time or failed. The first lead point
    /// past the finish line wins; at the time limit the rest rank by distance. Results go to
    /// the model (HUD), to MessagePipe, and to Logs/&lt;prefix&gt;_&lt;stamp&gt;.json after every race.
    ///
    /// No racer is ever stood up, re-seated or rescued (AGENTS rule H). A fallen racer keeps
    /// its own policy and gets up by itself or lies there; the report says it fell.
    /// </summary>
    public sealed class CreatureRaceSystem : IStartable, IFixedTickable, IDisposable
    {
        private const int FRAMES_TO_COMPILE_MUJOCO = 2;
        private const int FRAMES_BETWEEN_RACES = 2;
        private const int MAX_FRAMES_WAITING_FOR_PHYSICS = 120;
        private const float WATCHDOG_MARGIN_SECONDS = 15f;
        private const float TIMESTEP_TOLERANCE = 1e-4f;
        private const int VECTOR_SIZE = 3;

        private readonly CreatureRaceConfig _config;
        private readonly CreatureRaceModel _model;
        private readonly ICreatureSpawner _spawner;
        private readonly IPublisher<CreatureCountdownMessage> _countdownPublisher;
        private readonly IPublisher<CreatureRaceFinishedMessage> _raceFinishedPublisher;
        private readonly IPublisher<CreatureSeriesFinishedMessage> _seriesFinishedPublisher;
        private readonly CancellationTokenSource _cts = new();
        private readonly int _racerCount;
        private readonly CreaturePilot[] _pilots;
        private readonly CreaturePolicy[] _policies;
        private readonly int[] _ranking;

        private CreatureRaceReport.Series _series;
        private string _stamp = string.Empty;
        private bool _disposed;

        public CreatureRaceSystem(CreatureRaceConfig config, CreatureRaceModel model, ICreatureSpawner spawner,
                                  IPublisher<CreatureCountdownMessage> countdownPublisher,
                                  IPublisher<CreatureRaceFinishedMessage> raceFinishedPublisher,
                                  IPublisher<CreatureSeriesFinishedMessage> seriesFinishedPublisher)
        {
            _config = config;
            _model = model;
            _spawner = spawner;
            _countdownPublisher = countdownPublisher;
            _raceFinishedPublisher = raceFinishedPublisher;
            _seriesFinishedPublisher = seriesFinishedPublisher;
            // The model was sized from the same list; take the smaller in case they differ.
            _racerCount = Mathf.Min(model.Racers.Count, config.Racers.Count);
            _pilots = new CreaturePilot[_racerCount];
            _policies = new CreaturePolicy[_racerCount];
            _ranking = new int[_racerCount];
        }

        public void Start()
        {
            ConfigureRacerModels();
            _model.StartFocus = new Vector3(0f, 0f, _config.StartLineZ);
            _model.TrackLength = _config.TrackLength;
            _model.TimeLimitSeconds = _config.TimeLimitSeconds;

            if (!CreatureRaceRequest.TryConsume(out string mode, out float amount))
            {
                if (!_config.AutoStartSeries)
                {
                    _model.Phase = CreatureRacePhase.Idle;
                    _model.Message = "Auto-start is off. Start a series from the editor harness.";
                    return;
                }
                mode = CreatureRaceRequest.RACE_MODE;
                amount = _config.SeriesLength;
            }
            RunGuarded(mode, amount, _cts.Token).Forget();
        }

        /// <summary>Mirrors each pilot's telemetry into the model the HUD and camera read.</summary>
        public void FixedTick()
        {
            if (!_model.RacersSpawned)
            {
                return;
            }
            float raceClock = 0f;
            for (int lane = 0; lane < _racerCount; lane++)
            {
                CreaturePilot pilot = _pilots[lane];
                if (pilot == null)
                {
                    continue;
                }
                CreatureRacerModel racer = _model.Racers[lane];
                racer.Status = pilot.Status;
                racer.Distance = pilot.IsDone ? pilot.ResultDistance : pilot.Distance;
                racer.Speed = pilot.Speed;
                racer.ElapsedSeconds = pilot.ElapsedSeconds;
                racer.FinishTimeSeconds = pilot.FinishTimeSeconds;
                racer.Nose = pilot.Lead;
                racer.Upright = pilot.ReferenceUpright;
                racer.IsDown = _config.IsFallen(pilot.ReferenceUpright);
                if (pilot.Status != CreatureRacerStatus.Failed && pilot.ElapsedSeconds > raceClock)
                {
                    raceClock = pilot.ElapsedSeconds;
                }
            }
            if (_model.Phase == CreatureRacePhase.Racing || _model.Phase == CreatureRacePhase.SelfTest)
            {
                _model.ElapsedSeconds = raceClock;
            }
        }

        public void Dispose()
        {
            _disposed = true;
            _cts.Cancel();
            _cts.Dispose();
            for (int lane = 0; lane < _racerCount; lane++)
            {
                _policies[lane]?.Dispose();
                _policies[lane] = null;
            }
        }

        // ------------------------------------------------------------------ flow --

        private async UniTaskVoid RunGuarded(string mode, float amount, CancellationToken token)
        {
            bool isRace = CreatureRaceRequest.IsRace(mode);
            CreatureSelfTestDefinition test = isRace ? null : _config.FindSelfTest(mode);
            _model.IsRaceMode = isRace;
            _model.ModeLabel = isRace ? CreatureRaceRequest.RACE_MODE : test != null ? test.Name : mode;
            try
            {
                if (!isRace && test == null)
                {
                    Abort(mode, $"unknown self-test '{mode}'. This scene has: {SelfTestNames()}.");
                    return;
                }
                if (!TryPrepare(out CreatureLayout layout, out string error))
                {
                    Abort(_model.ModeLabel, error);
                    return;
                }
                if (isRace)
                {
                    await RunSeries(layout, Mathf.Max(1, Mathf.RoundToInt(amount)), token);
                }
                else
                {
                    await RunSelfTest(layout, test, amount, token);
                }
            }
            catch (OperationCanceledException)
            {
                // Play mode ended or the scope went away; the last per-race report stands.
            }
            catch (Exception exception)
            {
                if (_disposed)
                {
                    // Torn down mid-await (play mode ended): not a race error, and the
                    // per-race report on disk must not be overwritten with one.
                    return;
                }
                Debug.LogException(exception);
                Abort(_model.ModeLabel, exception.GetType().Name + ": " + exception.Message);
            }
        }

        private bool TryPrepare(out CreatureLayout layout, out string error)
        {
            if (!_spawner.TryPrepare(out layout, out error))
            {
                return false;
            }
            if (_racerCount == 0)
            {
                error = "the settings have no racers. Add them to the Racers list of the race settings asset.";
                return false;
            }
            if (Mathf.Abs(Time.fixedDeltaTime - layout.Rig.PhysicsDt) > TIMESTEP_TOLERANCE)
            {
                Debug.LogError($"[CreatureRace] Time.fixedDeltaTime is {Time.fixedDeltaTime:F4} s but the racers "
                             + $"trained at {layout.Rig.PhysicsDt:F3} s with decimation {layout.Rig.Decimation}. "
                             + "They will race at the wrong control rate; restore the project timestep.");
            }
            if (_pilots[0] == null)
            {
                for (int lane = 0; lane < _racerCount; lane++)
                {
                    CreatureRacerDefinition definition = _config.Racers[lane];
                    CreaturePolicy policy = CreaturePolicy.Create(definition.Brain, definition.Name,
                                                                  _config.ExpectedBrainPath(definition),
                                                                  layout.ObservationSize, layout.ActionSize);
                    _policies[lane] = policy;
                    _pilots[lane] = new CreaturePilot(policy, definition.Name, _config.PreviousActionClipped, layout);
                    CreatureRacerModel racer = _model.Racers[lane];
                    racer.BrainReady = policy.IsReady;
                    racer.BrainError = policy.Error;
                    if (!policy.IsReady)
                    {
                        Debug.LogError("[CreatureRace] " + policy.Error + " It will hold its rest pose and lose on distance.");
                    }
                }
            }
            error = string.Empty;
            return true;
        }

        private async UniTask RunSeries(CreatureLayout layout, int raceCount, CancellationToken token)
        {
            _model.ResetSeries();
            _model.PlannedRaces = raceCount;
            _stamp = CreatureReportWriter.Stamp();
            _series = NewSeriesReport(layout, raceCount);

            for (int raceNumber = 1; raceNumber <= raceCount; raceNumber++)
            {
                if (raceNumber > 1)
                {
                    // Results hold, then one clean frame gap between teardown and respawn.
                    await UniTask.Delay(TimeSpan.FromSeconds(_config.ResultsHoldSeconds), cancellationToken: token);
                    _spawner.Despawn();
                    _model.RacersSpawned = false;
                    await UniTask.DelayFrame(FRAMES_BETWEEN_RACES, PlayerLoopTiming.Update, token);
                }

                _model.RaceNumber = raceNumber;
                _model.Message = $"Race {raceNumber} of {raceCount}";
                await SpawnGrid(layout, _config.TrackLength, _config.TimeLimitSeconds, token);
                await Countdown(token);
                await WaitForPhysics(token);
                ReleaseAll();
                _model.Phase = CreatureRacePhase.Racing;
                await WaitUntilDecided(_config.TimeLimitSeconds, token);
                ConcludeRace(raceNumber);
            }

            _series.finishedAt = CreatureReportWriter.Now();
            WriteSeries();
            _model.Phase = CreatureRacePhase.SeriesComplete;
            _model.Message = SeriesTally();
            _seriesFinishedPublisher.Publish(new CreatureSeriesFinishedMessage(_model.ReportPath));
        }

        private async UniTask SpawnGrid(CreatureLayout layout, float finishDistance, float timeLimit,
                                        CancellationToken token)
        {
            _model.Phase = CreatureRacePhase.Spawning;
            _model.ElapsedSeconds = 0f;
            _model.CountdownValue = 0;
            float physicsDt = Time.fixedDeltaTime;
            for (int lane = 0; lane < _racerCount; lane++)
            {
                _pilots[lane].ResetForRace();
                _pilots[lane].Configure(_config.LaneOrigin(lane), Vector3.forward, finishDistance, timeLimit, physicsDt);
                _model.Racers[lane].ResetForRace();
            }

            if (!_spawner.MujocoSupported)
            {
                for (int lane = 0; lane < _racerCount; lane++)
                {
                    if (_config.Racers[lane].Physics == CreaturePhysicsKind.MujocoPlugin)
                    {
                        _pilots[lane].Fail(
                            $"{_model.Racers[lane].Name}: MuJoCo runs only on Windows in this project "
                          + "(Packages/org.mujoco ships mujoco.dll only, AGENTS rule F).");
                    }
                }
            }
            _spawner.Spawn(layout, _pilots);
            _model.RacersSpawned = true;

            // MjScene compiles in its Start, at the top of the next frame.
            await UniTask.DelayFrame(FRAMES_TO_COMPILE_MUJOCO, PlayerLoopTiming.Update, token);
        }

        private async UniTask Countdown(CancellationToken token)
        {
            _model.Phase = CreatureRacePhase.Countdown;
            for (int value = _config.CountdownSeconds; value > 0; value--)
            {
                _model.CountdownValue = value;
                _countdownPublisher.Publish(new CreatureCountdownMessage(value));
                await UniTask.Delay(TimeSpan.FromSeconds(1), cancellationToken: token);
            }
            _model.CountdownValue = 0;
        }

        /// <summary>
        /// Every racer's simulator must have stepped at least once before GO. A MuJoCo racer
        /// whose model failed to compile never gets a control callback; it is failed here, with
        /// a pointer to the console, instead of hanging the race.
        /// </summary>
        private async UniTask WaitForPhysics(CancellationToken token)
        {
            for (int frame = 0; frame < MAX_FRAMES_WAITING_FOR_PHYSICS && !AllPhysicsReady(); frame++)
            {
                await UniTask.Yield(PlayerLoopTiming.Update, token);
            }
            for (int lane = 0; lane < _racerCount; lane++)
            {
                CreaturePilot pilot = _pilots[lane];
                if (!pilot.PhysicsReady && pilot.Status != CreatureRacerStatus.Failed)
                {
                    pilot.Fail($"{_model.Racers[lane].Name}: its simulator never stepped - see the console "
                             + "for the error MjScene or the articulation build raised.");
                }
            }
        }

        private void ReleaseAll()
        {
            for (int lane = 0; lane < _racerCount; lane++)
            {
                _pilots[lane].Release();
            }
            _countdownPublisher.Publish(new CreatureCountdownMessage(0));
        }

        private async UniTask WaitUntilDecided(float raceSeconds, CancellationToken token)
        {
            float deadline = Time.time + raceSeconds + WATCHDOG_MARGIN_SECONDS;
            while (!AllDone())
            {
                if (Time.time > deadline)
                {
                    FailStragglers("watchdog: its race clock stopped advancing");
                    break;
                }
                await UniTask.Yield(PlayerLoopTiming.Update, token);
            }
            // One last copy, so the model shows the deciding step and not the one before.
            FixedTick();
        }

        // ------------------------------------------------------------- results --

        private void ConcludeRace(int raceNumber)
        {
            RankLanes();
            var race = new CreatureRaceReport.Race { raceNumber = raceNumber };
            bool anyFinished = false;
            bool anyTimedOut = false;
            for (int place = 0; place < _racerCount; place++)
            {
                int lane = _ranking[place];
                CreaturePilot pilot = _pilots[lane];
                CreatureRacerModel racer = _model.Racers[lane];
                racer.Place = place + 1;
                racer.AverageSpeed = AverageSpeed(pilot);
                anyFinished |= pilot.Status == CreatureRacerStatus.Finished;
                anyTimedOut |= pilot.Status == CreatureRacerStatus.TimedOut;
                race.racers.Add(new CreatureRaceReport.RacerResult
                {
                    lane = lane,
                    name = racer.Name,
                    method = racer.Method,
                    physics = racer.Physics,
                    brain = racer.BrainName,
                    brainLoaded = racer.BrainReady,
                    brainError = racer.BrainError,
                    status = pilot.Status.ToString(),
                    place = place + 1,
                    finishTimeSeconds = pilot.Status == CreatureRacerStatus.Finished ? pilot.FinishTimeSeconds : -1f,
                    distanceMeters = pilot.ResultDistance,
                    averageSpeedMps = racer.AverageSpeed,
                    minUpright = pilot.MinUprightSinceRelease,
                    fellOver = _config.IsFallen(pilot.MinUprightSinceRelease),
                    failReason = pilot.FailReason,
                });
            }

            int winner = _pilots[_ranking[0]].Status == CreatureRacerStatus.Failed ? -1 : _ranking[0];
            string endReason = anyFinished ? "finish" : anyTimedOut ? "timeLimit" : "failed";
            race.endReason = endReason;
            race.winner = winner >= 0 ? _model.Racers[winner].Name : "none";
            race.winnerMethod = winner >= 0 ? _model.Racers[winner].Method : "none";

            _model.LastWinnerLane = winner;
            _model.LastEndReason = endReason;
            _model.AddWin(winner);
            _model.CompletedRaces = raceNumber;
            _model.Phase = CreatureRacePhase.Results;
            _model.Message = winner >= 0
                ? $"Race {raceNumber}: {race.winner} wins ({endReason})"
                : $"Race {raceNumber}: no winner, every racer failed";

            _series.races.Add(race);
            _series.completedRaces = raceNumber;
            RebuildSummary();
            WriteSeries();
            _raceFinishedPublisher.Publish(new CreatureRaceFinishedMessage(raceNumber, winner));
        }

        /// <summary>Finished by time, then timed-out by distance, then failed by distance.</summary>
        private void RankLanes()
        {
            for (int lane = 0; lane < _racerCount; lane++)
            {
                _ranking[lane] = lane;
            }
            for (int index = 1; index < _racerCount; index++)
            {
                int current = _ranking[index];
                int slot = index - 1;
                while (slot >= 0 && CompareLanes(_ranking[slot], current) > 0)
                {
                    _ranking[slot + 1] = _ranking[slot];
                    slot--;
                }
                _ranking[slot + 1] = current;
            }
        }

        private int CompareLanes(int laneA, int laneB)
        {
            CreaturePilot pilotA = _pilots[laneA];
            CreaturePilot pilotB = _pilots[laneB];
            int groupA = StatusGroup(pilotA.Status);
            int groupB = StatusGroup(pilotB.Status);
            if (groupA != groupB)
            {
                return groupA.CompareTo(groupB);
            }
            if (pilotA.Status == CreatureRacerStatus.Finished)
            {
                return pilotA.FinishTimeSeconds.CompareTo(pilotB.FinishTimeSeconds);
            }
            return pilotB.ResultDistance.CompareTo(pilotA.ResultDistance);
        }

        private static int StatusGroup(CreatureRacerStatus status)
        {
            switch (status)
            {
                case CreatureRacerStatus.Finished:
                    return 0;
                case CreatureRacerStatus.Failed:
                    return 2;
                default:
                    return 1;
            }
        }

        private float AverageSpeed(CreaturePilot pilot)
        {
            switch (pilot.Status)
            {
                case CreatureRacerStatus.Finished:
                    return pilot.FinishTimeSeconds > 0f ? pilot.ResultDistance / pilot.FinishTimeSeconds : 0f;
                case CreatureRacerStatus.TimedOut:
                    return pilot.ResultDistance / _config.TimeLimitSeconds;
                default:
                    return pilot.ElapsedSeconds > 0f ? pilot.ResultDistance / pilot.ElapsedSeconds : 0f;
            }
        }

        private void RebuildSummary()
        {
            _series.summary.Clear();
            for (int lane = 0; lane < _racerCount; lane++)
            {
                CreatureRacerModel racer = _model.Racers[lane];
                var summary = new CreatureRaceReport.RacerSummary
                {
                    lane = lane,
                    name = racer.Name,
                    method = racer.Method,
                    physics = racer.Physics,
                    brain = racer.BrainName,
                    brainLoaded = racer.BrainReady,
                    brainError = racer.BrainError,
                    wins = _model.WinsFor(lane),
                };
                float timeSum = 0f;
                float distanceSum = 0f;
                float speedSum = 0f;
                int raceCount = 0;
                for (int raceIndex = 0; raceIndex < _series.races.Count; raceIndex++)
                {
                    CreatureRaceReport.Race race = _series.races[raceIndex];
                    for (int entryIndex = 0; entryIndex < race.racers.Count; entryIndex++)
                    {
                        CreatureRaceReport.RacerResult result = race.racers[entryIndex];
                        if (result.lane != lane)
                        {
                            continue;
                        }
                        raceCount++;
                        distanceSum += result.distanceMeters;
                        speedSum += result.averageSpeedMps;
                        if (result.fellOver)
                        {
                            summary.falls++;
                        }
                        if (result.finishTimeSeconds >= 0f)
                        {
                            summary.finishes++;
                            timeSum += result.finishTimeSeconds;
                            if (summary.bestFinishTimeSeconds < 0f || result.finishTimeSeconds < summary.bestFinishTimeSeconds)
                            {
                                summary.bestFinishTimeSeconds = result.finishTimeSeconds;
                            }
                        }
                    }
                }
                if (summary.finishes > 0)
                {
                    summary.meanFinishTimeSeconds = timeSum / summary.finishes;
                }
                if (raceCount > 0)
                {
                    summary.meanDistanceMeters = distanceSum / raceCount;
                    summary.meanAverageSpeedMps = speedSum / raceCount;
                }
                _series.summary.Add(summary);
            }
        }

        private void WriteSeries()
        {
            try
            {
                _model.ReportPath = CreatureReportWriter.Write(_series, _config.ReportPrefix + "_" + _stamp);
            }
            catch (Exception exception)
            {
                Debug.LogError("[CreatureRace] could not write the results file: " + exception.Message);
            }
        }

        private CreatureRaceReport.Series NewSeriesReport(CreatureLayout layout, int raceCount)
        {
            return new CreatureRaceReport.Series
            {
                kind = _config.ReportPrefix + "_series",
                creature = layout != null ? layout.Rig.Name : string.Empty,
                scene = SceneManager.GetActiveScene().path,
                startedAt = CreatureReportWriter.Now(),
                plannedRaces = raceCount,
                trackLengthMeters = _config.TrackLength,
                timeLimitSeconds = _config.TimeLimitSeconds,
                physicsDtSeconds = Time.fixedDeltaTime,
                finishRule = _config.FinishRule,
            };
        }

        // ----------------------------------------------------------- self-tests --

        private async UniTask RunSelfTest(CreatureLayout layout, CreatureSelfTestDefinition test, float seconds,
                                          CancellationToken token)
        {
            if (seconds <= 0f)
            {
                seconds = Mathf.Max(0.1f, test.DefaultSeconds);
            }
            if (!TryResolveTest(layout, test, out int joint, out int bodyA, out int bodyB, out string error))
            {
                Abort(test.Name, error);
                return;
            }
            _model.ResetSeries();
            string startedAt = CreatureReportWriter.Now();

            for (int lane = 0; lane < _racerCount; lane++)
            {
                _pilots[lane].SetProbe(bodyA, bodyB, test.ProbePoint);
            }
            await SpawnGrid(layout, float.PositiveInfinity, float.PositiveInfinity, token);
            _model.Phase = CreatureRacePhase.SelfTest;
            _model.Message = $"Self-test {test.Name}: settling";
            // Held at rest while every racer settles, exactly as before a race.
            await UniTask.Delay(TimeSpan.FromSeconds(_config.SelfTestSettleSeconds), cancellationToken: token);
            await WaitForPhysics(token);

            var action = new float[layout.ActionSize];
            if (joint >= 0)
            {
                action[joint] = test.TargetRad / layout.ActionScale;
            }
            for (int lane = 0; lane < _racerCount; lane++)
            {
                _pilots[lane].SetActionOverride(action);
                _pilots[lane].Release();
            }
            _model.Message = $"Self-test {test.Name}: running {seconds:0.#} s";

            float deadline = Time.time + seconds + WATCHDOG_MARGIN_SECONDS;
            while (!AllReached(seconds))
            {
                if (Time.time > deadline)
                {
                    FailStragglers("watchdog: its clock stopped advancing during the self-test");
                    break;
                }
                await UniTask.Yield(PlayerLoopTiming.Update, token);
            }
            FixedTick();

            CreatureRaceReport.SelfTest report = BuildSelfTestReport(layout, test, seconds, joint, startedAt);
            try
            {
                _model.ReportPath = CreatureReportWriter.Write(
                    report, $"{_config.ReportPrefix}_selftest_{test.Name}_{CreatureReportWriter.Stamp()}");
            }
            catch (Exception exception)
            {
                Debug.LogError("[CreatureRace] could not write the self-test file: " + exception.Message);
            }
            _model.Message = $"Self-test {test.Name}: {(report.allPassed ? "PASSED" : "FAILED")}";
            _model.Phase = CreatureRacePhase.SelfTestComplete;
            _seriesFinishedPublisher.Publish(new CreatureSeriesFinishedMessage(_model.ReportPath));
        }

        private static bool TryResolveTest(CreatureLayout layout, CreatureSelfTestDefinition test, out int joint,
                                           out int bodyA, out int bodyB, out string error)
        {
            joint = -1;
            bodyA = layout.LeadBody;
            bodyB = layout.LeadBody;
            error = string.Empty;
            if (test.Kind != CreatureSelfTestKind.JointSign)
            {
                return true;
            }
            joint = layout.Rig.FindAction(test.Joint);
            bodyA = layout.Rig.FindBody(test.ProbeBodyA);
            bodyB = layout.Rig.FindBody(test.ProbeBodyB);
            if (joint < 0 || bodyA < 0 || bodyB < 0)
            {
                error = $"self-test {test.Name}: joint '{test.Joint}' or probe bodies '{test.ProbeBodyA}', "
                      + $"'{test.ProbeBodyB}' are not in the rig";
                return false;
            }
            return true;
        }

        private CreatureRaceReport.SelfTest BuildSelfTestReport(CreatureLayout layout, CreatureSelfTestDefinition test,
                                                                float seconds, int joint, string startedAt)
        {
            var report = new CreatureRaceReport.SelfTest
            {
                kind = _config.ReportPrefix + "_selftest",
                creature = layout.Rig.Name,
                mode = test.Name,
                startedAt = startedAt,
                finishedAt = CreatureReportWriter.Now(),
                seconds = seconds,
                commandedJoint = joint >= 0 ? layout.Rig.ActionName(joint) : "none (all zero: the rest pose)",
                commandedRad = joint >= 0 ? test.TargetRad : 0f,
                expectation = test.Expectation,
                allPassed = true,
            };
            for (int lane = 0; lane < _racerCount; lane++)
            {
                CreaturePilot pilot = _pilots[lane];
                var finalJoints = new float[layout.ActionSize];
                for (int actionIndex = 0; actionIndex < layout.ActionSize; actionIndex++)
                {
                    finalJoints[actionIndex] = pilot.LastJointPosition(actionIndex);
                }
                Vector3 offset = pilot.ProbeOffset;
                Vector3 measured = test.ProbeRelativeToRelease ? offset - pilot.ProbeOffsetAtRelease : offset;
                var entry = new CreatureRaceReport.SelfTestRacer
                {
                    lane = lane,
                    name = _model.Racers[lane].Name,
                    method = _model.Racers[lane].Method,
                    physics = _model.Racers[lane].Physics,
                    brain = _model.Racers[lane].BrainName,
                    brainLoaded = _model.Racers[lane].BrainReady,
                    brainError = _model.Racers[lane].BrainError,
                    status = pilot.Status.ToString(),
                    commandedJointRad = joint >= 0 ? pilot.LastJointPosition(joint) : 0f,
                    maxAbsJointRad = pilot.MaxAbsJointSinceRelease,
                    probeOffsetMeters = new float[VECTOR_SIZE] { offset.x, offset.y, offset.z },
                    probeAlongAxisMeters = Vector3.Dot(measured, test.ProbeAxis.normalized),
                    leadDisplacementMeters = (pilot.Lead - pilot.LeadAtRelease).magnitude,
                    referenceHeightMeters = pilot.ReferenceHeight,
                    referenceUpright = pilot.ReferenceUpright,
                    finalSpeedMps = pilot.Speed,
                    finalJointRad = finalJoints,
                };
                entry.passed = Judge(layout, test, joint, entry, pilot, out string verdict);
                entry.verdict = verdict;
                report.allPassed &= entry.passed;
                report.racers.Add(entry);
            }
            return report;
        }

        private static bool Judge(CreatureLayout layout, CreatureSelfTestDefinition test, int joint,
                                  CreatureRaceReport.SelfTestRacer entry, CreaturePilot pilot, out string verdict)
        {
            if (pilot.Status == CreatureRacerStatus.Failed)
            {
                verdict = "failed: " + pilot.FailReason;
                return false;
            }
            if (test.Kind == CreatureSelfTestKind.JointSign)
            {
                bool angle = entry.commandedJointRad - layout.Rest(joint) > test.MinJointRad;
                bool probe = entry.probeAlongAxisMeters > test.MinProbeOffset;
                verdict = $"{layout.Rig.ActionName(joint)} positive={angle} {test.ProbeLabel}={probe}";
                return angle && probe;
            }

            bool still = entry.leadDisplacementMeters < test.MaxLeadDisplacement
                      && Mathf.Abs(entry.finalSpeedMps) < test.MaxSpeed;
            bool atRest = entry.maxAbsJointRad < test.MaxJointFromRest;
            bool atHeight = Mathf.Abs(entry.referenceHeightMeters - test.ExpectedReferenceHeight)
                          < test.ReferenceHeightTolerance;
            bool checkUpright = test.MinUpright > -1f;
            bool upright = !checkUpright || entry.referenceUpright >= test.MinUpright;
            verdict = $"still={still} atRestPose={atRest} atHeight={atHeight}"
                    + (checkUpright ? $" upright={upright}" : string.Empty);
            return still && atRest && atHeight && upright;
        }

        // -------------------------------------------------------------- helpers --

        private void ConfigureRacerModels()
        {
            for (int lane = 0; lane < _racerCount; lane++)
            {
                CreatureRacerDefinition definition = _config.Racers[lane];
                CreatureRacerModel racer = _model.Racers[lane];
                racer.Name = definition.Name;
                racer.Method = definition.Method;
                racer.Physics = definition.PhysicsLabel;
                racer.Color = definition.Color;
                racer.BrainName = definition.Brain != null ? definition.Brain.name : "missing";
                // Until TryPrepare loads it, so the HUD never shows a brain that is absent.
                racer.BrainReady = definition.Brain != null;
            }
        }

        /// <summary>"Series done: Quad (MuJoCo) 5 - Quad (Isaac Lab 3) 0".</summary>
        private string SeriesTally()
        {
            var text = new StringBuilder("Series done: ");
            for (int lane = 0; lane < _racerCount; lane++)
            {
                if (lane > 0)
                {
                    text.Append(" - ");
                }
                text.Append(_model.Racers[lane].Name).Append(' ').Append(_model.WinsFor(lane));
            }
            return text.ToString();
        }

        private string SelfTestNames()
        {
            var text = new StringBuilder();
            for (int index = 0; index < _config.SelfTests.Count; index++)
            {
                CreatureSelfTestDefinition test = _config.SelfTests[index];
                if (index > 0)
                {
                    text.Append(", ");
                }
                text.Append(test.Name).Append(" (\"").Append(test.Alias).Append("\")");
            }
            return text.Length > 0 ? text.ToString() : "none";
        }

        private bool AllPhysicsReady()
        {
            for (int lane = 0; lane < _racerCount; lane++)
            {
                CreaturePilot pilot = _pilots[lane];
                if (!pilot.PhysicsReady && pilot.Status != CreatureRacerStatus.Failed)
                {
                    return false;
                }
            }
            return true;
        }

        private bool AllDone()
        {
            for (int lane = 0; lane < _racerCount; lane++)
            {
                if (!_pilots[lane].IsDone)
                {
                    return false;
                }
            }
            return true;
        }

        private bool AllReached(float seconds)
        {
            for (int lane = 0; lane < _racerCount; lane++)
            {
                CreaturePilot pilot = _pilots[lane];
                if (pilot.Status != CreatureRacerStatus.Failed && pilot.ElapsedSeconds < seconds)
                {
                    return false;
                }
            }
            return true;
        }

        private void FailStragglers(string reason)
        {
            for (int lane = 0; lane < _racerCount; lane++)
            {
                CreaturePilot pilot = _pilots[lane];
                if (!pilot.IsDone)
                {
                    pilot.Fail($"{_model.Racers[lane].Name}: {reason}");
                }
            }
        }

        private void Abort(string mode, string error)
        {
            Debug.LogError("[CreatureRace] " + error);
            _spawner.Despawn();
            _model.RacersSpawned = false;
            _model.Phase = CreatureRacePhase.Error;
            _model.Message = error;
            try
            {
                if (CreatureRaceRequest.IsRace(mode))
                {
                    if (_series == null)
                    {
                        _stamp = CreatureReportWriter.Stamp();
                        _series = NewSeriesReport(null, 0);
                    }
                    _series.error = error;
                    _series.finishedAt = CreatureReportWriter.Now();
                    _model.ReportPath = CreatureReportWriter.Write(_series, _config.ReportPrefix + "_" + _stamp);
                }
                else
                {
                    var report = new CreatureRaceReport.SelfTest
                    {
                        kind = _config.ReportPrefix + "_selftest",
                        mode = mode,
                        startedAt = CreatureReportWriter.Now(),
                        finishedAt = CreatureReportWriter.Now(),
                        error = error,
                        allPassed = false,
                    };
                    _model.ReportPath = CreatureReportWriter.Write(
                        report, $"{_config.ReportPrefix}_selftest_{mode}_{CreatureReportWriter.Stamp()}");
                }
            }
            catch (Exception exception)
            {
                Debug.LogError("[CreatureRace] could not write the error report: " + exception.Message);
            }
            _seriesFinishedPublisher.Publish(new CreatureSeriesFinishedMessage(_model.ReportPath));
        }
    }
}
