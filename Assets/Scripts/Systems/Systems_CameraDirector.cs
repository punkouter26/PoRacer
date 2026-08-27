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
    /// Presentation referee. The pack camera frames the whole grid through the
    /// countdown; once racing starts the shot tightens onto whoever is in front
    /// and stays with the lead, handing over to a new leader as it changes.
    /// OrbitCameraView cuts between broadcast angles on that target.
    /// InputView's NextTarget/PrevTarget switch the orbit to a chosen racer
    /// manually, and Overview returns to the wide static shot. Both race rigs
    /// are built at runtime.
    /// </summary>
    public sealed class Systems_CameraDirector : ITickable, IDisposable
    {
        private const int ACTIVE_PRIORITY = 20;
        private const int INACTIVE_PRIORITY = 0;
        private const float NEAR_CLIP = 0.3f;
        private const float FAR_CLIP = 400f;
        private readonly RaceModel _model;
        private readonly CameraRigView _rig;
        private readonly List<Transform> _targets = new();
        private readonly Dictionary<string, Transform> _targetsByRacerId = new();
        private readonly IDisposable _subscription;
        private readonly IDisposable _raceFinishedSubscription;
        private int _targetIndex = -1;
        private CinemachineCamera _orbitCamera;
        private OrbitCameraView _orbit;
        private CinemachineCamera _packCamera;
        private PackCameraView _pack;
        private Transform _orbitTarget;
        private bool _followingLeader;
        private Bounds _keepOut;
        private bool _hasKeepOut;

        public Systems_CameraDirector(RaceModel model, CameraRigView rig,
            ISubscriber<LeadChangedMessage> leadChanged, ISubscriber<RaceFinishedMessage> raceFinished)
        {
            _model = model;
            _rig = rig;
            _subscription = leadChanged.Subscribe(OnLeadChanged);
            _raceFinishedSubscription = raceFinished.Subscribe(OnRaceFinished);
            ShowOverview();
        }

        /// <summary>
        /// Lead watcher: hands the shot to the orbit camera on the racer out in
        /// front and re-aims it whenever the lead changes, so the coverage
        /// follows the story rather than the whole field.
        /// </summary>
        public void Tick()
        {
            if (!_model.RaceActive)
            {
                return;
            }
            RacerState front = null;
            for (int racerIndex = 0; racerIndex < _model.Racers.Count; racerIndex++)
            {
                RacerState racer = _model.Racers[racerIndex];
                if (racer.Status == RacerStatus.Racing
                    && (front == null || racer.Progress > front.Progress))
                {
                    front = racer;
                }
            }
            if (front == null)
            {
                return;
            }
            // The leader owns the shot for the whole race, not just the finish.
            // Re-aiming only when the front runner actually changes keeps the
            // angle cycling in OrbitCameraView on its own clock instead of being
            // reset every frame.
            _followingLeader = true;
            if (_targetsByRacerId.TryGetValue(front.RacerId, out Transform target)
                && target != null && target.gameObject.activeInHierarchy
                && _orbitTarget != target)
            {
                OrbitAround(target);
            }
        }

        public void SetTargets(IReadOnlyList<Transform> targets)
        {
            _targets.Clear();
            _targetsByRacerId.Clear();
            for (int targetIndex = 0; targetIndex < targets.Count; targetIndex++)
            {
                _targets.Add(targets[targetIndex]);
            }
            _targetIndex = -1;
            _followingLeader = false;
            _orbitTarget = null;
            if (_targets.Count > 0)
            {
                // Race default: frame the whole field, not one star.
                EnsurePackCamera();
                _pack.SetTargets(_targets);
                ShowPack();
            }
            else
            {
                ShowOverview();
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
            _orbit?.SetKeepOut(keepOut);
        }

        public void ClearKeepOut()
        {
            _hasKeepOut = false;
            _orbit?.ClearKeepOut();
        }

        /// <summary>Spawn registers each racer so lead changes can find its transform.</summary>
        public void RegisterRacer(string racerId, Transform target)
        {
            _targetsByRacerId[racerId] = target;
        }

        private void OnLeadChanged(LeadChangedMessage message)
        {
            // The pack shot already contains the new leader; only re-aim if the
            // viewer is on the single-racer orbit right now.
            if (_orbitCamera == null || _orbitCamera.Priority != ACTIVE_PRIORITY)
            {
                return;
            }
            if (!_targetsByRacerId.TryGetValue(message.RacerId, out Transform leader)
                || leader == null || !leader.gameObject.activeInHierarchy)
            {
                return;
            }
            OrbitAround(leader);
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
                bestPlace = result.Place;
            }
            if (winner != null)
            {
                // Matches what the lead watcher would have left behind, so the
                // next SetTargets clears the same state either way.
                _followingLeader = true;
                OrbitAround(winner);
                return;
            }
            if (_targets.Count > 0)
            {
                ShowPack();
            }
        }

        public void NextTarget() => CycleTarget(1);

        /// <summary>
        /// The racer the shot is currently built around, or null on the wide
        /// overview and pack shots. Read by PostFxView to size the shadow range to
        /// what the camera is actually looking at.
        /// </summary>
        public Transform ActiveShotTarget =>
            _orbitCamera != null && _orbitCamera.Priority == ACTIVE_PRIORITY ? _orbitTarget : null;

        public void PrevTarget() => CycleTarget(-1);

        public void ShowOverview()
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

        public void Dispose()
        {
            _subscription?.Dispose();
            _raceFinishedSubscription?.Dispose();
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
            OrbitAround(target);
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
            if (_hasKeepOut)
            {
                _orbit.SetKeepOut(_keepOut);
            }
        }
    }
}
