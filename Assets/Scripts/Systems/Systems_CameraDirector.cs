using System;
using System.Collections.Generic;
using MessagePipe;
using PoRacer.Models;
using PoRacer.Views;
using Unity.Cinemachine;
using UnityEngine;
using VContainer.Unity;

namespace PoRacer.Systems
{
    /// <summary>
    /// Broadcast director. The pack camera frames the whole grid through the countdown;
    /// once racing starts, every frame each racer earns an interest score and the shot
    /// goes to whoever is the best story right now:
    ///
    ///   * the leader, and anyone close to the front;
    ///   * a battle — two racers within a couple of metres of each other get a duel shot
    ///     that frames both (a Cinemachine target group);
    ///   * a racer going down, getting itself back up, or just back on its feet — the
    ///     knockdown window of AGENTS rule H is the most dramatic thing in the sport, and
    ///     it used to happen off camera;
    ///   * the final metres before the line, and a fresh change of leader.
    ///
    /// Cuts are paced like sports coverage: a shot holds for MIN_SHOT_SECONDS before
    /// anything can take it, only a fall or the finish can cut sooner, and a subject held
    /// past MAX_SHOT_SECONDS loses its hold so the field gets seen. Scores come from
    /// RaceModel (progress, status) and RaceTelemetryModel (falling, getting up), both of
    /// which are read-only here.
    ///
    /// The viewer outranks the director. Picking a racer (swipe, keys or a tap on it)
    /// orbits it, opens its telemetry card and holds the auto-cuts off for
    /// VIEWER_HOLD_SECONDS; a tap on empty space asks for the wide shot the same way.
    /// OrbitCameraView still cuts between broadcast angles on whatever this picks.
    /// </summary>
    public sealed class Systems_CameraDirector : ITickable, IDisposable
    {
        private const int ACTIVE_PRIORITY = 20;
        private const int INACTIVE_PRIORITY = 0;
        private const float NEAR_CLIP = 0.3f;
        private const float FAR_CLIP = 400f;

        // --- Shot pacing (unscaled seconds, so a slow-motion finish does not stretch them) ---
        private const float MIN_SHOT_SECONDS = 3.5f;
        private const float MIN_URGENT_SHOT_SECONDS = 1.2f;
        private const float MAX_SHOT_SECONDS = 11f;
        private const float VIEWER_HOLD_SECONDS = 20f;
        // The start is worth watching whole: the pack shot keeps the field this long past GO.
        private const float PACK_AFTER_GO_SECONDS = 2.5f;
        // A new story on the same subject (the leader hits the final metres, a duelist
        // stumbles) updates the caption without a cut, but not more often than this.
        private const float MIN_CAPTION_SECONDS = 2f;

        // --- Interest weights. The leader scores LEAD_WEIGHT, second half that, and so on;
        // the other terms are sized against that so each wins only when it really is the story.
        private const float LEAD_WEIGHT = 1f;
        private const float BATTLE_WEIGHT = 0.9f;
        private const float DANGER_WEIGHT = 1.6f;
        private const float GETTING_UP_WEIGHT = 1.3f;
        private const float BACK_UP_WEIGHT = 1f;
        private const float FINISH_WEIGHT = 1.4f;
        private const float NEW_LEADER_WEIGHT = 0.8f;
        private const float STICKY_BONUS = 0.25f;
        private const float BOREDOM_PENALTY = 0.7f;
        // A racer lying still is not a story; one fighting to get up is.
        private const float LYING_STILL_PENALTY = 0.6f;
        // A cut only happens when the challenger beats the current subject by this much...
        private const float CUT_MARGIN = 0.15f;
        // ...and an early cut needs an urgent story and a clear lead over the current shot.
        private const float URGENT_MARGIN = 0.5f;

        // --- Story thresholds ---
        private const float DUEL_GAP_METERS = 2.5f;
        private const float FALL_RATE_THRESHOLD = 0.6f;
        private const float FALL_RATE_FULL = 2f;
        private const float BACK_UP_SECONDS = 3f;
        private const float FINISH_ZONE_METERS = 6f;
        private const float NEW_LEADER_SECONDS = 3f;
        // Duel framing: widen the orbit as the pair separates, and never past this.
        private const float DUEL_FRAMING_METERS_PER_STEP = 5f;
        private const float MAX_DUEL_FRAMING = 1.8f;
        private const float DUEL_MEMBER_RADIUS = 0.6f;

        // --- Tap-to-pick: how close to a racer's screen position a tap must land ---
        private const float PICK_RADIUS_WIDTH_FRACTION = 0.12f;

        private readonly RaceModel _model;
        private readonly RaceTelemetryModel _telemetry;
        private readonly CameraRigView _rig;
        private readonly IPublisher<CameraShotChangedMessage> _shotChangedPublisher;
        private readonly List<Transform> _targets = new();
        private readonly Dictionary<string, Transform> _targetsByRacerId = new();
        private readonly Dictionary<Transform, string> _racerIdsByTarget = new();
        private readonly IDisposable _subscription;
        private readonly IDisposable _raceFinishedSubscription;
        private int _targetIndex = -1;
        private CinemachineCamera _orbitCamera;
        private OrbitCameraView _orbit;
        private CinemachineCamera _packCamera;
        private PackCameraView _pack;
        private CinemachineTargetGroup _duelGroup;
        private Transform _orbitTarget;
        private Bounds _keepOut;
        private bool _hasKeepOut;
        private Systems_CoursePath _course;
        private Camera _outputCamera;

        // --- Current shot ---
        private string _subjectId;
        private string _rivalId;
        private ShotReason _captionReason = ShotReason.Pack;
        private float _captionAt;
        private float _shotStartedAt;
        private float _viewerHoldUntil;
        private string _newLeaderId;
        private float _newLeaderUntil;

        public Systems_CameraDirector(RaceModel model, CameraRigView rig, RaceTelemetryModel telemetry,
            ISubscriber<LeadChangedMessage> leadChanged, ISubscriber<RaceFinishedMessage> raceFinished,
            IPublisher<CameraShotChangedMessage> shotChangedPublisher = null)
        {
            _model = model;
            _rig = rig;
            _telemetry = telemetry;
            _shotChangedPublisher = shotChangedPublisher;
            _subscription = leadChanged.Subscribe(OnLeadChanged);
            _raceFinishedSubscription = raceFinished.Subscribe(OnRaceFinished);
            ApplyOverview();
        }

        /// <summary>
        /// The racer the shot is currently built around, or null on the wide overview and
        /// pack shots. Read by PostFxView to size the shadow range to what the camera is
        /// actually looking at; on a duel it is the pair's midpoint.
        /// </summary>
        public Transform ActiveShotTarget =>
            _orbitCamera != null && _orbitCamera.Priority == ACTIVE_PRIORITY ? _orbitTarget : null;

        public void Tick()
        {
            if (!_model.RaceActive)
            {
                return;
            }
            if (_rivalId != null)
            {
                UpdateDuelFraming();
            }
            float now = Time.unscaledTime;
            if (now < _viewerHoldUntil)
            {
                return;
            }
            if (_subjectId == null && _model.ElapsedSeconds < PACK_AFTER_GO_SECONDS)
            {
                return;
            }

            RacerState best = PickStory(now, out ShotReason reason, out RacerState rival, out float bestScore,
                out float currentScore);
            if (best == null)
            {
                return;
            }
            string rivalId = reason == ShotReason.Duel && rival != null ? rival.RacerId : null;
            bool sameShot = best.RacerId == _subjectId && rivalId == _rivalId;
            // The same pair seen from the other side is still the same duel.
            bool samePair = rivalId != null && best.RacerId == _rivalId && rivalId == _subjectId;
            // Same subject, different framing: the racer being followed has just got into
            // (or out of) a battle. Its score is the same either way, so the margins below
            // can never pick it; switch between solo and duel once the shot has matured.
            bool reframe = !sameShot && !samePair
                && (best.RacerId == _subjectId || best.RacerId == _rivalId)
                && (rivalId == null) != (_rivalId == null);
            if (reframe && now - _shotStartedAt >= MIN_SHOT_SECONDS)
            {
                CutTo(best.RacerId, rivalId, reason, now);
                return;
            }
            if (sameShot || samePair || reframe)
            {
                if (reason != _captionReason && reason != ShotReason.Leader
                    && now - _captionAt >= MIN_CAPTION_SECONDS)
                {
                    Publish(reason);
                }
                return;
            }

            float shotAge = now - _shotStartedAt;
            // The subject is out of the race (finished, DNF) or there is none yet: move on now.
            bool currentGone = currentScore == float.NegativeInfinity;
            bool urgent = (reason == ShotReason.GoingDown || reason == ShotReason.FinalMetres)
                && bestScore > currentScore + URGENT_MARGIN;
            bool matured = shotAge >= MIN_SHOT_SECONDS && bestScore > currentScore + CUT_MARGIN;
            if (currentGone || matured || (urgent && shotAge >= MIN_URGENT_SHOT_SECONDS))
            {
                CutTo(best.RacerId, rivalId, reason, now);
            }
        }

        public void SetTargets(IReadOnlyList<Transform> targets)
        {
            _targets.Clear();
            _targetsByRacerId.Clear();
            _racerIdsByTarget.Clear();
            for (int targetIndex = 0; targetIndex < targets.Count; targetIndex++)
            {
                _targets.Add(targets[targetIndex]);
            }
            _targetIndex = -1;
            _orbitTarget = null;
            _subjectId = null;
            _rivalId = null;
            _viewerHoldUntil = 0f;
            _newLeaderId = null;
            _telemetry.FocusedRacerId = null;
            if (_targets.Count > 0)
            {
                // Race default: frame the whole field, not one star.
                EnsurePackCamera();
                _pack.SetTargets(_targets);
                ShowPack();
                _shotStartedAt = Time.unscaledTime;
            }
            else
            {
                ApplyOverview();
            }
        }

        /// <summary>
        /// The course being raced, or null on builder maps. The pack camera frames
        /// along it; the orbit camera places its shots on it, so that coverage of a
        /// course runs down the road instead of through whatever the map is built
        /// inside. Held, because both rigs are created lazily and a course set
        /// before the orbit exists must still reach it.
        /// </summary>
        public void SetCourse(Systems_CoursePath course)
        {
            _course = course;
            if (_pack != null)
            {
                _pack.SetCourse(course);
            }
            if (_orbit != null)
            {
                _orbit.SetCourse(course);
            }
        }

        /// <summary>
        /// Spawn hands over the finish arch's volume after each track build. The
        /// arch is collider-free decoration, so nothing else stops the orbit shot
        /// from sweeping straight into it as the leader crosses the line.
        /// </summary>
        public void SetKeepOut(Bounds keepOut)
        {
            _keepOut = keepOut;
            _hasKeepOut = true;
            if (_orbit != null)
            {
                _orbit.SetKeepOut(keepOut);
            }
        }

        public void ClearKeepOut()
        {
            _hasKeepOut = false;
            if (_orbit != null)
            {
                _orbit.ClearKeepOut();
            }
        }

        /// <summary>Spawn registers each racer so the director can find its transform, and back.</summary>
        public void RegisterRacer(string racerId, Transform target)
        {
            _targetsByRacerId[racerId] = target;
            if (target != null)
            {
                _racerIdsByTarget[target] = racerId;
            }
        }

        public void NextTarget() => CycleTarget(1);

        public void PrevTarget() => CycleTarget(-1);

        /// <summary>
        /// The viewer asked for the wide shot. Holds the auto-cuts off like any viewer
        /// choice, and closes the telemetry card.
        /// </summary>
        public void ShowOverview()
        {
            _viewerHoldUntil = Time.unscaledTime + VIEWER_HOLD_SECONDS;
            _telemetry.FocusedRacerId = null;
            _subjectId = null;
            _rivalId = null;
            ApplyOverview();
            Publish(ShotReason.Overview);
        }

        /// <summary>
        /// Hands the camera back to the director — the telemetry card's close button. The
        /// next tick cuts to the best story straight away.
        /// </summary>
        public void ResumeDirecting()
        {
            _viewerHoldUntil = 0f;
            _telemetry.FocusedRacerId = null;
            _shotStartedAt = float.NegativeInfinity;
        }

        /// <summary>
        /// A tap: picks the racer nearest the tap on screen, if one is close enough.
        /// Returns false when the tap landed on empty track, which the caller treats as a
        /// request for the wide shot.
        /// </summary>
        public bool TryPickAt(Vector2 screenPoint)
        {
            if (_outputCamera == null)
            {
                _outputCamera = Camera.main;
            }
            if (_outputCamera == null || _targets.Count == 0)
            {
                return false;
            }
            float pickRadius = Screen.width * PICK_RADIUS_WIDTH_FRACTION;
            float bestDistance = pickRadius * pickRadius;
            int bestIndex = -1;
            for (int targetIndex = 0; targetIndex < _targets.Count; targetIndex++)
            {
                Transform target = _targets[targetIndex];
                if (target == null || !target.gameObject.activeInHierarchy)
                {
                    continue;
                }
                Vector3 onScreen = _outputCamera.WorldToScreenPoint(target.position);
                if (onScreen.z <= 0f)
                {
                    continue;
                }
                float distance = ((Vector2)onScreen - screenPoint).sqrMagnitude;
                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    bestIndex = targetIndex;
                }
            }
            if (bestIndex < 0)
            {
                return false;
            }
            _targetIndex = bestIndex;
            FocusViewerPick(_targets[bestIndex]);
            return true;
        }

        public void Dispose()
        {
            _subscription?.Dispose();
            _raceFinishedSubscription?.Dispose();
        }

        /// <summary>
        /// Scores every racer still racing and returns the best story. Also reports the
        /// current subject's score, or negative infinity when the current subject is no
        /// longer racing (DNF, finished) and the shot must move on regardless.
        /// </summary>
        private RacerState PickStory(float now, out ShotReason bestReason, out RacerState bestRival,
            out float bestScore, out float currentScore)
        {
            RacerState best = null;
            bestReason = ShotReason.Leader;
            bestRival = null;
            bestScore = float.NegativeInfinity;
            currentScore = float.NegativeInfinity;
            float shotAge = now - _shotStartedAt;
            float trackLength = Mathf.Max(1f, _model.TrackLengthMeters);

            for (int racerIndex = 0; racerIndex < _model.Racers.Count; racerIndex++)
            {
                RacerState racer = _model.Racers[racerIndex];
                if (racer.Status != RacerStatus.Racing || !HasLiveTarget(racer.RacerId))
                {
                    continue;
                }

                int rank = 0;
                RacerState rival = null;
                float nearestGap = float.PositiveInfinity;
                for (int otherIndex = 0; otherIndex < _model.Racers.Count; otherIndex++)
                {
                    RacerState other = _model.Racers[otherIndex];
                    if (other == racer || other.Status != RacerStatus.Racing)
                    {
                        continue;
                    }
                    if (other.Progress > racer.Progress)
                    {
                        rank++;
                    }
                    float gap = Mathf.Abs(other.Progress - racer.Progress);
                    if (gap < nearestGap && HasLiveTarget(other.RacerId))
                    {
                        nearestGap = gap;
                        rival = other;
                    }
                }

                ShotReason reason = ShotReason.Leader;
                float lead = LEAD_WEIGHT / (1 + rank);
                float strongest = lead;
                float score = lead;

                float battle = nearestGap < DUEL_GAP_METERS
                    ? BATTLE_WEIGHT * (1f - nearestGap / DUEL_GAP_METERS) * (rank < 3 ? 1f : 0.6f)
                    : 0f;
                score += battle;
                if (battle > strongest)
                {
                    strongest = battle;
                    reason = ShotReason.Duel;
                }

                RacerTelemetry telemetry = _telemetry.Find(racer.RacerId);
                if (telemetry != null)
                {
                    float drama = 0f;
                    ShotReason dramaReason = reason;
                    if (!telemetry.IsDown && telemetry.FallRate > FALL_RATE_THRESHOLD)
                    {
                        drama = DANGER_WEIGHT * Mathf.Clamp01(telemetry.FallRate / FALL_RATE_FULL);
                        dramaReason = ShotReason.GoingDown;
                    }
                    else if (telemetry.IsGettingUp)
                    {
                        drama = GETTING_UP_WEIGHT;
                        dramaReason = ShotReason.GettingUp;
                    }
                    else if (now - telemetry.RecoveredAt < BACK_UP_SECONDS)
                    {
                        drama = BACK_UP_WEIGHT;
                        dramaReason = ShotReason.BackUp;
                    }
                    else if (telemetry.IsDown)
                    {
                        score -= LYING_STILL_PENALTY;
                    }
                    score += drama;
                    if (drama > strongest)
                    {
                        strongest = drama;
                        reason = dramaReason;
                    }
                }

                float remaining = trackLength - racer.Progress;
                float finish = remaining < FINISH_ZONE_METERS
                    ? FINISH_WEIGHT * (1f - Mathf.Max(0f, remaining) / FINISH_ZONE_METERS)
                    : 0f;
                score += finish;
                if (finish > strongest)
                {
                    strongest = finish;
                    reason = ShotReason.FinalMetres;
                }

                // Only while it really is in front: the lead watcher's cooldown can leave a
                // "new leader" standing for a racer that has already been passed again.
                if (racer.RacerId == _newLeaderId && now < _newLeaderUntil && rank == 0)
                {
                    score += NEW_LEADER_WEIGHT;
                    if (NEW_LEADER_WEIGHT > strongest)
                    {
                        reason = ShotReason.NewLeader;
                    }
                }

                bool isSubject = racer.RacerId == _subjectId || racer.RacerId == _rivalId;
                if (isSubject)
                {
                    score += shotAge < MAX_SHOT_SECONDS ? STICKY_BONUS : -BOREDOM_PENALTY;
                    currentScore = Mathf.Max(currentScore, score);
                }

                if (score > bestScore)
                {
                    best = racer;
                    bestScore = score;
                    bestReason = reason;
                    bestRival = reason == ShotReason.Duel ? rival : null;
                }
            }
            return best;
        }

        private bool HasLiveTarget(string racerId)
        {
            return _targetsByRacerId.TryGetValue(racerId, out Transform target)
                && target != null && target.gameObject.activeInHierarchy;
        }

        private void CutTo(string subjectId, string rivalId, ShotReason reason, float now)
        {
            if (!_targetsByRacerId.TryGetValue(subjectId, out Transform subject))
            {
                return;
            }
            _subjectId = subjectId;
            _rivalId = rivalId;
            _shotStartedAt = now;
            if (rivalId != null && _targetsByRacerId.TryGetValue(rivalId, out Transform rival))
            {
                EnsureDuelGroup();
                _duelGroup.Targets.Clear();
                _duelGroup.AddMember(subject, 1f, DUEL_MEMBER_RADIUS);
                _duelGroup.AddMember(rival, 1f, DUEL_MEMBER_RADIUS);
                // Place the group before the orbit reads it, or the first frame of the
                // duel frames wherever the group was left by the last one.
                _duelGroup.transform.position = (subject.position + rival.position) * 0.5f;
                OrbitAround(_duelGroup.transform);
                UpdateDuelFraming();
            }
            else
            {
                _rivalId = null;
                OrbitAround(subject);
                _orbit.SetFramingScale(1f);
            }
            Publish(reason);
        }

        /// <summary>Widens the duel shot as the pair drifts apart, so both stay in frame.</summary>
        private void UpdateDuelFraming()
        {
            if (_orbit == null
                || !_targetsByRacerId.TryGetValue(_subjectId, out Transform subject)
                || !_targetsByRacerId.TryGetValue(_rivalId, out Transform rival)
                || subject == null || rival == null)
            {
                return;
            }
            float separation = Vector3.Distance(subject.position, rival.position);
            _orbit.SetFramingScale(Mathf.Clamp(1f + separation / DUEL_FRAMING_METERS_PER_STEP, 1f, MAX_DUEL_FRAMING));
        }

        private void Publish(ShotReason reason)
        {
            _captionReason = reason;
            _captionAt = Time.unscaledTime;
            _shotChangedPublisher?.Publish(new CameraShotChangedMessage(reason, _subjectId, _rivalId));
        }

        private void OnLeadChanged(LeadChangedMessage message)
        {
            // A new leader is a story for a few seconds; the scoring decides whether it
            // is the best one on track right now.
            _newLeaderId = message.RacerId;
            _newLeaderUntil = Time.unscaledTime + NEW_LEADER_SECONDS;
        }

        /// <summary>
        /// Race over: hold the shot on the winner while the results panel is up.
        /// Tick() stops steering the moment RaceActive clears, so without this the
        /// camera freezes on whatever it happened to be showing — and in an
        /// all-DNF field, where no racer was ever "in front" and racing, that is
        /// the wide pack shot of a pile-up.
        /// Results arrive in grid order rather than finishing order, so this scans
        /// for the best placed racer that still has a live transform to frame; if
        /// none survives, fall back to the field.
        /// </summary>
        private void OnRaceFinished(RaceFinishedMessage message)
        {
            Transform winner = null;
            string winnerId = null;
            int bestPlace = int.MaxValue;
            for (int resultIndex = 0; resultIndex < message.Results.Count; resultIndex++)
            {
                RaceResultEntry result = message.Results[resultIndex];
                if (result.Place <= 0 || result.Place >= bestPlace)
                {
                    continue;
                }
                if (!_targetsByRacerId.TryGetValue(result.RacerId, out Transform target)
                    || target == null || !target.gameObject.activeInHierarchy)
                {
                    continue;
                }
                winner = target;
                winnerId = result.RacerId;
                bestPlace = result.Place;
            }
            _viewerHoldUntil = 0f;
            _telemetry.FocusedRacerId = null;
            if (winner != null)
            {
                _subjectId = winnerId;
                _rivalId = null;
                OrbitAround(winner);
                _orbit.SetFramingScale(1f);
                Publish(ShotReason.Winner);
                return;
            }
            if (_targets.Count > 0)
            {
                ShowPack();
            }
        }

        private void CycleTarget(int direction)
        {
            if (_targets.Count == 0)
            {
                return;
            }
            _targetIndex = (_targetIndex + direction + _targets.Count) % _targets.Count;
            Transform target = _targets[_targetIndex];
            if (target == null)
            {
                return;
            }
            FocusViewerPick(target);
        }

        /// <summary>
        /// The viewer chose this racer: orbit it, open its telemetry card, and keep the
        /// director's hands off for a while.
        /// </summary>
        private void FocusViewerPick(Transform target)
        {
            _racerIdsByTarget.TryGetValue(target, out string racerId);
            _viewerHoldUntil = Time.unscaledTime + VIEWER_HOLD_SECONDS;
            _subjectId = racerId;
            _rivalId = null;
            _shotStartedAt = Time.unscaledTime;
            _telemetry.FocusedRacerId = racerId;
            OrbitAround(target);
            _orbit.SetFramingScale(1f);
            Publish(ShotReason.ViewerPick);
        }

        private void ApplyOverview()
        {
            _rig.OverviewCamera.Priority = ACTIVE_PRIORITY;
            if (_orbitCamera != null)
            {
                _orbitCamera.Priority = INACTIVE_PRIORITY;
            }
            if (_packCamera != null)
            {
                _packCamera.Priority = INACTIVE_PRIORITY;
            }
        }

        private void ShowPack()
        {
            _packCamera.Priority = ACTIVE_PRIORITY;
            _rig.OverviewCamera.Priority = INACTIVE_PRIORITY;
            if (_orbitCamera != null)
            {
                _orbitCamera.Priority = INACTIVE_PRIORITY;
            }
        }

        private void OrbitAround(Transform target)
        {
            EnsureOrbitCamera();
            _orbitTarget = target;
            _orbit.SetTarget(target);
            _orbitCamera.Priority = ACTIVE_PRIORITY;
            _rig.OverviewCamera.Priority = INACTIVE_PRIORITY;
            if (_packCamera != null)
            {
                _packCamera.Priority = INACTIVE_PRIORITY;
            }
        }

        /// <summary>
        /// The duel shot's subject: a target group that keeps its own transform at the
        /// pair's midpoint, which the orbit rig then circles like any single racer.
        /// </summary>
        private void EnsureDuelGroup()
        {
            if (_duelGroup != null)
            {
                return;
            }
            var go = new GameObject("CM_DuelGroup");
            go.transform.SetParent(_rig.transform, false);
            _duelGroup = go.AddComponent<CinemachineTargetGroup>();
            _duelGroup.PositionMode = CinemachineTargetGroup.PositionModes.GroupCenter;
        }

        /// <summary>
        /// Builds the pack rig on first use: a passive CinemachineCamera whose
        /// transform is driven by PackCameraView, plus an impulse listener so
        /// camera shake still lands on the pack shot.
        /// </summary>
        private void EnsurePackCamera()
        {
            if (_packCamera != null)
            {
                return;
            }
            var go = new GameObject("CM_Pack");
            go.transform.SetParent(_rig.transform, false);
            _packCamera = go.AddComponent<CinemachineCamera>();
            LensSettings lens = _packCamera.Lens;
            lens.NearClipPlane = NEAR_CLIP;
            lens.FarClipPlane = FAR_CLIP;
            _packCamera.Lens = lens;
            _packCamera.Priority = INACTIVE_PRIORITY;
            go.AddComponent<CinemachineImpulseListener>();
            _pack = go.AddComponent<PackCameraView>();
            // Same replay as the orbit rig below, and for the same reason: SetCourse runs
            // once per race from the spawner, and on the first race of a session it lands
            // before this rig exists. Without this the pack shot frames the first course
            // race along world +Z instead of down the road.
            _pack.SetCourse(_course);
            // The lens the pack camera adapts for portrait is this one, not Camera.main:
            // CinemachineBrain overwrites the Camera's own field of view every frame.
            _pack.BindLens(_packCamera);
        }

        /// <summary>
        /// Builds the orbit rig on first use: a passive CinemachineCamera whose
        /// transform is driven by OrbitCameraView, plus an impulse listener so
        /// camera shake still lands on the orbit shot.
        /// </summary>
        private void EnsureOrbitCamera()
        {
            if (_orbitCamera != null)
            {
                return;
            }
            var go = new GameObject("CM_Orbit");
            go.transform.SetParent(_rig.transform, false);
            _orbitCamera = go.AddComponent<CinemachineCamera>();
            LensSettings lens = _orbitCamera.Lens;
            lens.NearClipPlane = NEAR_CLIP;
            lens.FarClipPlane = FAR_CLIP;
            _orbitCamera.Lens = lens;
            _orbitCamera.Priority = INACTIVE_PRIORITY;
            go.AddComponent<CinemachineImpulseListener>();
            _orbit = go.AddComponent<OrbitCameraView>();
            // The lens the orbit shot adapts for portrait is this one, not Camera.main:
            // CinemachineBrain overwrites the Camera's field of view every frame. Same
            // binding the pack rig gets above, and for the same reason — without it the
            // shot that owns almost the whole race runs at an unadapted 40 deg while the
            // pack shot widens, and the subject crops on anything but 9:16.
            _orbit.BindLens(_orbitCamera);
            if (_hasKeepOut)
            {
                _orbit.SetKeepOut(_keepOut);
            }
            // Spawn sets the course once per race, which can land before this rig
            // exists; replay it so course coverage is not lost to creation order.
            _orbit.SetCourse(_course);
        }
    }
}
