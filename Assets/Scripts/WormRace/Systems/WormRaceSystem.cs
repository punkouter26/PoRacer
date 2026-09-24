using System;
using System.Text;
using System.Threading;
using Cysharp.Threading.Tasks;
using MessagePipe;
using UnityEngine;
using UnityEngine.SceneManagement;
using VContainer.Unity;

namespace PoRacer.WormRace
{
    /// <summary>
    /// Runs SCN_WORM_RACE: a series of N races (or one self-test), start to finish.
    ///
    /// One racer per entry of WormRaceSettings' racer list (lane = index), each with its own
    /// pilot and brain; nothing here knows which lane runs in which simulator.
    ///
    /// Per race: spawn every worm straight on the start line, a 3-2-1 countdown with every
    /// pilot held (physics runs, so all worms settle onto the floor the same way; the
    /// policies do not), GO, then wait until each worm has finished, run out of time or
    /// failed. The first nose past the finish line wins; at the time limit the rest rank
    /// by distance. Results go to the model (HUD), to MessagePipe, and to
    /// Logs/wormrace_&lt;stamp&gt;.json after every race, so a series cut short still leaves
    /// everything up to its last completed race on disk.
    ///
    /// No racer is ever stood up, re-seated or rescued (AGENTS rule H). The worm cannot
    /// really fall, but a diverged or failed one stays where it is until teardown.
    /// </summary>
    public sealed class WormRaceSystem : IStartable, IFixedTickable, IDisposable
    {
        private const int FRAMES_TO_COMPILE_MUJOCO = 2;
        private const int FRAMES_BETWEEN_RACES = 2;
        private const int MAX_FRAMES_WAITING_FOR_PHYSICS = 120;
        private const float WATCHDOG_MARGIN_SECONDS = 15f;
        private const float TIMESTEP_TOLERANCE = 1e-4f;

        private const float SIGN_TEST_TARGET_RAD = 0.5f;
        private const float DEFAULT_ZERO_TEST_SECONDS = 5f;
        private const float DEFAULT_SIGN_TEST_SECONDS = 2f;
        private const float ZERO_TEST_MAX_DISPLACEMENT = 0.02f;
        private const float ZERO_TEST_MAX_JOINT_RAD = 0.05f;
        private const float ZERO_TEST_HEIGHT_TOLERANCE = 0.01f;
        private const float ZERO_TEST_MAX_SPEED = 0.01f;
        private const float SIGN_TEST_MIN_ANGLE_RAD = 0.3f;
        private const float SIGN_TEST_MIN_OFFSET = 0.02f;

        private readonly WormRaceSettings _settings;
        private readonly WormRaceModel _model;
        private readonly WormSpawnSystem _spawn;
        private readonly IPublisher<WormCountdownMessage> _countdownPublisher;
        private readonly IPublisher<WormRaceFinishedMessage> _raceFinishedPublisher;
        private readonly IPublisher<WormSeriesFinishedMessage> _seriesFinishedPublisher;
        private readonly CancellationTokenSource _cts = new();
        private readonly int _racerCount;
        private readonly WormPilot[] _pilots;
        private readonly WormPolicy[] _policies;
        private readonly int[] _ranking;

        private WormRaceReport.Series _series;
        private string _stamp = string.Empty;
        private bool _disposed;

        public WormRaceSystem(WormRaceSettings settings, WormRaceModel model, WormSpawnSystem spawn,
                              IPublisher<WormCountdownMessage> countdownPublisher,
                              IPublisher<WormRaceFinishedMessage> raceFinishedPublisher,
                              IPublisher<WormSeriesFinishedMessage> seriesFinishedPublisher)
        {
            _settings = settings;
            _model = model;
            _spawn = spawn;
            _countdownPublisher = countdownPublisher;
            _raceFinishedPublisher = raceFinishedPublisher;
            _seriesFinishedPublisher = seriesFinishedPublisher;
            // The model was sized from the same list; take the smaller in case they differ.
            _racerCount = Mathf.Min(model.Racers.Count, settings.Racers.Count);
            _pilots = new WormPilot[_racerCount];
            _policies = new WormPolicy[_racerCount];
            _ranking = new int[_racerCount];
        }

        public void Start()
        {
            ConfigureRacerModels();
            _model.StartFocus = new Vector3(0f, 0f, _settings.StartLineZ);
            _model.TrackLength = _settings.TrackLength;
            _model.TimeLimitSeconds = _settings.TimeLimitSeconds;

            if (!WormRaceRequest.TryConsume(out WormRaceMode mode, out float amount))
            {
                if (!_settings.AutoStartSeries)
                {
                    _model.Phase = WormRacePhase.Idle;
                    _model.Message = "Auto-start is off. Run PoRacer.WormRace.EditorTools.Editor_WormRace.Start().";
                    return;
                }
                mode = WormRaceMode.Race;
                amount = _settings.SeriesLength;
            }
            RunGuarded(mode, amount, _cts.Token).Forget();
        }

        /// <summary>Mirrors each pilot's telemetry into the model the HUD and camera read.</summary>
        public void FixedTick()
        {
            if (!_model.WormsSpawned)
            {
                return;
            }
            float raceClock = 0f;
            for (int lane = 0; lane < _racerCount; lane++)
            {
                WormPilot pilot = _pilots[lane];
                if (pilot == null)
                {
                    continue;
                }
                WormRacerModel racer = _model.Racers[lane];
                racer.Status = pilot.Status;
                racer.Distance = pilot.IsDone ? pilot.ResultDistance : pilot.Distance;
                racer.Speed = pilot.Speed;
                racer.ElapsedSeconds = pilot.ElapsedSeconds;
                racer.FinishTimeSeconds = pilot.FinishTimeSeconds;
                racer.Nose = pilot.Nose;
                if (pilot.Status != WormRacerStatus.Failed && pilot.ElapsedSeconds > raceClock)
                {
                    raceClock = pilot.ElapsedSeconds;
                }
            }
            if (_model.Phase == WormRacePhase.Racing || _model.Phase == WormRacePhase.SelfTest)
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

        private async UniTaskVoid RunGuarded(WormRaceMode mode, float amount, CancellationToken token)
        {
            _model.Mode = mode;
            try
            {
                if (!TryPrepare(out WormRig rig, out string error))
                {
                    Abort(mode, error);
                    return;
                }
                if (mode == WormRaceMode.Race)
                {
                    await RunSeries(rig, Mathf.Max(1, Mathf.RoundToInt(amount)), token);
                }
                else
                {
                    await RunSelfTest(rig, mode, amount, token);
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
                Abort(mode, exception.GetType().Name + ": " + exception.Message);
            }
        }

        private bool TryPrepare(out WormRig rig, out string error)
        {
            if (!_spawn.TryGetRig(out rig, out error))
            {
                return false;
            }
            if (_racerCount == 0)
            {
                error = "WormRaceSettings has no racers. Run "
                      + "PoRacer.WormRace.EditorTools.Editor_BuildWormRaceScene.Build() to seed the racer list.";
                return false;
            }
            if (Mathf.Abs(Time.fixedDeltaTime - WormContract.PHYSICS_DT) > TIMESTEP_TOLERANCE)
            {
                Debug.LogError($"[WormRace] Time.fixedDeltaTime is {Time.fixedDeltaTime:F4} s but the worms "
                             + $"trained at {WormContract.PHYSICS_DT:F3} s with decimation {WormContract.DECIMATION}. "
                             + "They will race at the wrong control rate; restore the project timestep.");
            }
            if (_pilots[0] == null)
            {
                for (int lane = 0; lane < _racerCount; lane++)
                {
                    WormRacerDefinition definition = _settings.Racers[lane];
                    WormPolicy policy = WormPolicy.Create(definition.Brain, definition.Name,
                                                          definition.ExpectedBrainPath);
                    _policies[lane] = policy;
                    _pilots[lane] = new WormPilot(policy, definition.Name, _settings.PreviousActionClipped);
                    WormRacerModel racer = _model.Racers[lane];
                    racer.BrainReady = policy.IsReady;
                    racer.BrainError = policy.Error;
                    if (!policy.IsReady)
                    {
                        Debug.LogError("[WormRace] " + policy.Error + " It will lie still and lose on distance.");
                    }
                }
            }
            error = string.Empty;
            return true;
        }

        private async UniTask RunSeries(WormRig rig, int raceCount, CancellationToken token)
        {
            _model.ResetSeries();
            _model.PlannedRaces = raceCount;
            _stamp = WormReportWriter.Stamp();
            _series = NewSeriesReport(raceCount);

            for (int raceNumber = 1; raceNumber <= raceCount; raceNumber++)
            {
                if (raceNumber > 1)
                {
                    // Results hold, then one clean frame gap between teardown and respawn.
                    await UniTask.Delay(TimeSpan.FromSeconds(_settings.ResultsHoldSeconds), cancellationToken: token);
                    _spawn.Despawn();
                    _model.WormsSpawned = false;
                    await UniTask.DelayFrame(FRAMES_BETWEEN_RACES, PlayerLoopTiming.Update, token);
                }

                _model.RaceNumber = raceNumber;
                _model.Message = $"Race {raceNumber} of {raceCount}";
                await SpawnGrid(rig, _settings.TrackLength, _settings.TimeLimitSeconds, token);
                await Countdown(token);
                await WaitForPhysics(token);
                ReleaseAll();
                _model.Phase = WormRacePhase.Racing;
                await WaitUntilDecided(_settings.TimeLimitSeconds, token);
                ConcludeRace(raceNumber);
            }

            _series.finishedAt = WormReportWriter.Now();
            WriteSeries();
            _model.Phase = WormRacePhase.SeriesComplete;
            _model.Message = SeriesTally();
            _seriesFinishedPublisher.Publish(new WormSeriesFinishedMessage(_model.ReportPath));
        }

        private async UniTask SpawnGrid(WormRig rig, float finishDistance, float timeLimit, CancellationToken token)
        {
            _model.Phase = WormRacePhase.Spawning;
            _model.ElapsedSeconds = 0f;
            _model.CountdownValue = 0;
            float physicsDt = Time.fixedDeltaTime;
            for (int lane = 0; lane < _racerCount; lane++)
            {
                _pilots[lane].ResetForRace();
                _pilots[lane].Configure(_spawn.LaneOrigin(lane), Vector3.forward, finishDistance, timeLimit, physicsDt);
                _model.Racers[lane].ResetForRace();
            }

            if (!_spawn.MujocoSupported)
            {
                for (int lane = 0; lane < _racerCount; lane++)
                {
                    if (_settings.Racers[lane].Physics == WormPhysicsKind.MujocoPlugin)
                    {
                        _pilots[lane].Fail(
                            $"{_model.Racers[lane].Name}: MuJoCo runs only on Windows in this project "
                          + "(Packages/org.mujoco ships mujoco.dll only, AGENTS rule F).");
                    }
                }
            }
            _spawn.Spawn(rig, _pilots);
            _model.WormsSpawned = true;

            // MjScene compiles in its Start, at the top of the next frame.
            await UniTask.DelayFrame(FRAMES_TO_COMPILE_MUJOCO, PlayerLoopTiming.Update, token);
        }

        private async UniTask Countdown(CancellationToken token)
        {
            _model.Phase = WormRacePhase.Countdown;
            for (int value = _settings.CountdownSeconds; value > 0; value--)
            {
                _model.CountdownValue = value;
                _countdownPublisher.Publish(new WormCountdownMessage(value));
                await UniTask.Delay(TimeSpan.FromSeconds(1), cancellationToken: token);
            }
            _model.CountdownValue = 0;
        }

        /// <summary>
        /// Every racer's simulator must have stepped at least once before GO. A MuJoCo worm whose
        /// model failed to compile never gets a control callback; it is failed here, with a
        /// pointer to the console, instead of hanging the race.
        /// </summary>
        private async UniTask WaitForPhysics(CancellationToken token)
        {
            for (int frame = 0; frame < MAX_FRAMES_WAITING_FOR_PHYSICS && !AllPhysicsReady(); frame++)
            {
                await UniTask.Yield(PlayerLoopTiming.Update, token);
            }
            for (int lane = 0; lane < _racerCount; lane++)
            {
                WormPilot pilot = _pilots[lane];
                if (!pilot.PhysicsReady && pilot.Status != WormRacerStatus.Failed)
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
            _countdownPublisher.Publish(new WormCountdownMessage(0));
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
            var race = new WormRaceReport.Race { raceNumber = raceNumber };
            bool anyFinished = false;
            bool anyTimedOut = false;
            for (int place = 0; place < _racerCount; place++)
            {
                int lane = _ranking[place];
                WormPilot pilot = _pilots[lane];
                WormRacerModel racer = _model.Racers[lane];
                racer.Place = place + 1;
                racer.AverageSpeed = AverageSpeed(pilot);
                anyFinished |= pilot.Status == WormRacerStatus.Finished;
                anyTimedOut |= pilot.Status == WormRacerStatus.TimedOut;
                race.racers.Add(new WormRaceReport.RacerResult
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
                    finishTimeSeconds = pilot.Status == WormRacerStatus.Finished ? pilot.FinishTimeSeconds : -1f,
                    distanceMeters = pilot.ResultDistance,
                    averageSpeedMps = racer.AverageSpeed,
                    failReason = pilot.FailReason,
                });
            }

            int winner = _pilots[_ranking[0]].Status == WormRacerStatus.Failed ? -1 : _ranking[0];
            string endReason = anyFinished ? "finish" : anyTimedOut ? "timeLimit" : "failed";
            race.endReason = endReason;
            race.winner = winner >= 0 ? _model.Racers[winner].Name : "none";
            race.winnerMethod = winner >= 0 ? _model.Racers[winner].Method : "none";

            _model.LastWinnerLane = winner;
            _model.LastEndReason = endReason;
            _model.AddWin(winner);
            _model.CompletedRaces = raceNumber;
            _model.Phase = WormRacePhase.Results;
            _model.Message = winner >= 0
                ? $"Race {raceNumber}: {race.winner} wins ({endReason})"
                : $"Race {raceNumber}: no winner, every racer failed";

            _series.races.Add(race);
            _series.completedRaces = raceNumber;
            RebuildSummary();
            WriteSeries();
            _raceFinishedPublisher.Publish(new WormRaceFinishedMessage(raceNumber, winner));
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
            WormPilot pilotA = _pilots[laneA];
            WormPilot pilotB = _pilots[laneB];
            int groupA = StatusGroup(pilotA.Status);
            int groupB = StatusGroup(pilotB.Status);
            if (groupA != groupB)
            {
                return groupA.CompareTo(groupB);
            }
            if (pilotA.Status == WormRacerStatus.Finished)
            {
                return pilotA.FinishTimeSeconds.CompareTo(pilotB.FinishTimeSeconds);
            }
            return pilotB.ResultDistance.CompareTo(pilotA.ResultDistance);
        }

        private static int StatusGroup(WormRacerStatus status)
        {
            switch (status)
            {
                case WormRacerStatus.Finished:
                    return 0;
                case WormRacerStatus.Failed:
                    return 2;
                default:
                    return 1;
            }
        }

        private float AverageSpeed(WormPilot pilot)
        {
            switch (pilot.Status)
            {
                case WormRacerStatus.Finished:
                    return pilot.FinishTimeSeconds > 0f ? pilot.ResultDistance / pilot.FinishTimeSeconds : 0f;
                case WormRacerStatus.TimedOut:
                    return pilot.ResultDistance / _settings.TimeLimitSeconds;
                default:
                    return pilot.ElapsedSeconds > 0f ? pilot.ResultDistance / pilot.ElapsedSeconds : 0f;
            }
        }

        private void RebuildSummary()
        {
            _series.summary.Clear();
            for (int lane = 0; lane < _racerCount; lane++)
            {
                WormRacerModel racer = _model.Racers[lane];
                var summary = new WormRaceReport.RacerSummary
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
                    WormRaceReport.Race race = _series.races[raceIndex];
                    for (int entryIndex = 0; entryIndex < race.racers.Count; entryIndex++)
                    {
                        WormRaceReport.RacerResult result = race.racers[entryIndex];
                        if (result.lane != lane)
                        {
                            continue;
                        }
                        raceCount++;
                        distanceSum += result.distanceMeters;
                        speedSum += result.averageSpeedMps;
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
                _model.ReportPath = WormReportWriter.Write(_series, "wormrace_" + _stamp);
            }
            catch (Exception exception)
            {
                Debug.LogError("[WormRace] could not write the results file: " + exception.Message);
            }
        }

        private WormRaceReport.Series NewSeriesReport(int raceCount)
        {
            return new WormRaceReport.Series
            {
                scene = SceneManager.GetActiveScene().path,
                startedAt = WormReportWriter.Now(),
                plannedRaces = raceCount,
                trackLengthMeters = _settings.TrackLength,
                timeLimitSeconds = _settings.TimeLimitSeconds,
                physicsDtSeconds = Time.fixedDeltaTime,
            };
        }

        // ----------------------------------------------------------- self-tests --

        private async UniTask RunSelfTest(WormRig rig, WormRaceMode mode, float seconds, CancellationToken token)
        {
            if (seconds <= 0f)
            {
                seconds = mode == WormRaceMode.ZeroActionTest ? DEFAULT_ZERO_TEST_SECONDS : DEFAULT_SIGN_TEST_SECONDS;
            }
            _model.ResetSeries();
            string startedAt = WormReportWriter.Now();

            await SpawnGrid(rig, float.PositiveInfinity, float.PositiveInfinity, token);
            _model.Phase = WormRacePhase.SelfTest;
            _model.Message = $"Self-test {mode}: settling";
            // Held straight while every worm settles onto the floor, exactly as before a race.
            await UniTask.Delay(TimeSpan.FromSeconds(_settings.SelfTestSettleSeconds), cancellationToken: token);
            await WaitForPhysics(token);

            int joint = -1;
            if (mode == WormRaceMode.YawSignTest)
            {
                joint = WormContract.J0_YAW_INDEX;
            }
            else if (mode == WormRaceMode.PitchSignTest)
            {
                joint = WormContract.J0_PITCH_INDEX;
            }
            var action = new float[WormContract.ACTION_SIZE];
            if (joint >= 0)
            {
                action[joint] = SIGN_TEST_TARGET_RAD / WormContract.JOINT_RANGE_RAD;
            }
            for (int lane = 0; lane < _racerCount; lane++)
            {
                _pilots[lane].SetActionOverride(action);
                _pilots[lane].Release();
            }
            _model.Message = $"Self-test {mode}: running {seconds:0.#} s";

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

            WormRaceReport.SelfTest report = BuildSelfTestReport(rig, mode, seconds, joint, startedAt);
            try
            {
                _model.ReportPath = WormReportWriter.Write(
                    report, $"wormrace_selftest_{mode}_{WormReportWriter.Stamp()}");
            }
            catch (Exception exception)
            {
                Debug.LogError("[WormRace] could not write the self-test file: " + exception.Message);
            }
            _model.Message = $"Self-test {mode}: {(report.allPassed ? "PASSED" : "FAILED")}";
            _model.Phase = WormRacePhase.SelfTestComplete;
            _seriesFinishedPublisher.Publish(new WormSeriesFinishedMessage(_model.ReportPath));
        }

        private WormRaceReport.SelfTest BuildSelfTestReport(WormRig rig, WormRaceMode mode, float seconds,
                                                            int joint, string startedAt)
        {
            var report = new WormRaceReport.SelfTest
            {
                mode = mode.ToString(),
                startedAt = startedAt,
                finishedAt = WormReportWriter.Now(),
                seconds = seconds,
                commandedJoint = joint >= 0 ? WormContract.ActionOrder[joint] : "none (all zero)",
                commandedRad = joint >= 0 ? SIGN_TEST_TARGET_RAD : 0f,
                expectation = Expectation(mode),
                allPassed = true,
            };
            for (int lane = 0; lane < _racerCount; lane++)
            {
                WormPilot pilot = _pilots[lane];
                var finalJoints = new float[WormContract.ACTION_SIZE];
                for (int actionIndex = 0; actionIndex < WormContract.ACTION_SIZE; actionIndex++)
                {
                    finalJoints[actionIndex] = pilot.LastJointPosition(actionIndex);
                }
                var entry = new WormRaceReport.SelfTestRacer
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
                    secondSegmentLateralMeters = pilot.SecondSegmentLateral,
                    secondSegmentVerticalMeters = pilot.SecondSegmentVertical,
                    noseDisplacementMeters = (pilot.Nose - pilot.NoseAtRelease).magnitude,
                    referenceSegmentHeightMeters = pilot.ReferenceHeight,
                    finalSpeedMps = pilot.Speed,
                    finalJointRad = finalJoints,
                };
                entry.passed = Judge(mode, entry, pilot, rig.SegmentRadius, out string verdict);
                entry.verdict = verdict;
                report.allPassed &= entry.passed;
                report.racers.Add(entry);
            }
            return report;
        }

        private static bool Judge(WormRaceMode mode, WormRaceReport.SelfTestRacer entry, WormPilot pilot,
                                  float radius, out string verdict)
        {
            if (pilot.Status == WormRacerStatus.Failed)
            {
                verdict = "failed: " + pilot.FailReason;
                return false;
            }
            switch (mode)
            {
                case WormRaceMode.ZeroActionTest:
                {
                    bool still = entry.noseDisplacementMeters < ZERO_TEST_MAX_DISPLACEMENT
                              && Mathf.Abs(entry.finalSpeedMps) < ZERO_TEST_MAX_SPEED;
                    bool straight = entry.maxAbsJointRad < ZERO_TEST_MAX_JOINT_RAD;
                    bool onFloor = Mathf.Abs(entry.referenceSegmentHeightMeters - radius) < ZERO_TEST_HEIGHT_TOLERANCE;
                    verdict = $"still={still} straight={straight} onFloor={onFloor}";
                    return still && straight && onFloor;
                }
                case WormRaceMode.YawSignTest:
                {
                    bool angle = entry.commandedJointRad > SIGN_TEST_MIN_ANGLE_RAD;
                    bool side = entry.secondSegmentLateralMeters > SIGN_TEST_MIN_OFFSET;
                    verdict = $"j0_yaw positive={angle} segment1 to the worm's right (Unity +x)={side}";
                    return angle && side;
                }
                case WormRaceMode.PitchSignTest:
                {
                    bool angle = entry.commandedJointRad > SIGN_TEST_MIN_ANGLE_RAD;
                    bool up = entry.secondSegmentVerticalMeters > SIGN_TEST_MIN_OFFSET;
                    verdict = $"j0_pitch positive={angle} segment1 above the head's plane={up}";
                    return angle && up;
                }
                default:
                    verdict = "not a self-test mode";
                    return false;
            }
        }

        private static string Expectation(WormRaceMode mode)
        {
            switch (mode)
            {
                case WormRaceMode.ZeroActionTest:
                    return "zero action: nose moves < 2 cm, speed < 1 cm/s, every |joint| < 0.05 rad, "
                         + "segment 2 centre at the capsule radius (0.045 m) +/- 1 cm, for every racer in its own physics";
                case WormRaceMode.YawSignTest:
                    return "MuJoCo j0_yaw = +0.5 rad swings segment 1 toward MuJoCo -y = Unity +x (the worm's "
                         + "right when facing +Z): j0_yaw reads > +0.3 rad and segment 1 sits > 2 cm to the "
                         + "right of the head, for every racer in its own physics";
                case WormRaceMode.PitchSignTest:
                    return "MuJoCo j0_pitch = +0.5 rad rotates segment 1 about +y, lifting it relative to the "
                         + "head: j0_pitch reads > +0.3 rad and segment 1 sits > 2 cm above the head's "
                         + "plane, for every racer in its own physics";
                default:
                    return string.Empty;
            }
        }

        // -------------------------------------------------------------- helpers --

        private void ConfigureRacerModels()
        {
            for (int lane = 0; lane < _racerCount; lane++)
            {
                WormRacerDefinition definition = _settings.Racers[lane];
                WormRacerModel racer = _model.Racers[lane];
                racer.Name = definition.Name;
                racer.Method = definition.Method;
                racer.Physics = definition.PhysicsLabel;
                racer.Color = definition.Color;
                racer.BrainName = definition.Brain != null ? definition.Brain.name : "missing";
                // Until TryPrepare loads it, so the HUD never shows a brain that is absent.
                racer.BrainReady = definition.Brain != null;
            }
        }

        /// <summary>"Series done: MuJoCo worm 5 - Isaac worm 0 - Isaac Lab 3 worm 0".</summary>
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

        private bool AllPhysicsReady()
        {
            for (int lane = 0; lane < _racerCount; lane++)
            {
                WormPilot pilot = _pilots[lane];
                if (!pilot.PhysicsReady && pilot.Status != WormRacerStatus.Failed)
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
                WormPilot pilot = _pilots[lane];
                if (pilot.Status != WormRacerStatus.Failed && pilot.ElapsedSeconds < seconds)
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
                WormPilot pilot = _pilots[lane];
                if (!pilot.IsDone)
                {
                    pilot.Fail($"{_model.Racers[lane].Name}: {reason}");
                }
            }
        }

        private void Abort(WormRaceMode mode, string error)
        {
            Debug.LogError("[WormRace] " + error);
            _spawn.Despawn();
            _model.WormsSpawned = false;
            _model.Phase = WormRacePhase.Error;
            _model.Message = error;
            try
            {
                if (mode == WormRaceMode.Race)
                {
                    if (_series == null)
                    {
                        _stamp = WormReportWriter.Stamp();
                        _series = NewSeriesReport(0);
                    }
                    _series.error = error;
                    _series.finishedAt = WormReportWriter.Now();
                    _model.ReportPath = WormReportWriter.Write(_series, "wormrace_" + _stamp);
                }
                else
                {
                    var report = new WormRaceReport.SelfTest
                    {
                        mode = mode.ToString(),
                        startedAt = WormReportWriter.Now(),
                        finishedAt = WormReportWriter.Now(),
                        error = error,
                        allPassed = false,
                    };
                    _model.ReportPath = WormReportWriter.Write(
                        report, $"wormrace_selftest_{mode}_{WormReportWriter.Stamp()}");
                }
            }
            catch (Exception exception)
            {
                Debug.LogError("[WormRace] could not write the error report: " + exception.Message);
            }
            _seriesFinishedPublisher.Publish(new WormSeriesFinishedMessage(_model.ReportPath));
        }
    }
}
