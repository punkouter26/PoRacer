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
    /// Presentation referee. The default race shot is a pack camera that frames
    /// every active racer at once; once the front runner enters the final
    /// stretch the shot tightens to an orbit around them, so the finish plays
    /// on the star. InputView's NextTarget/PrevTarget switch to the orbit
    /// manually, and Overview returns to the wide static shot. Both race rigs
    /// are built at runtime.
    /// </summary>
    public sealed class Systems_CameraDirector : ITickable, IDisposable
    {
        private const int ACTIVE_PRIORITY = 20;
        private const int INACTIVE_PRIORITY = 0;
        private const float NEAR_CLIP = 0.3f;
        private const float FAR_CLIP = 400f;
        // Once the front runner passes this fraction of the track, feature them.
        // Late on purpose: the wide pack shot owns most of the race, the
        // close-up only owns the finish itself.
        private const float FINAL_STRETCH_FRACTION = 0.9f;

        private readonly RaceModel _model;
        private readonly CameraRigView _rig;
        private readonly List<Transform> _targets = new();
        private readonly Dictionary<string, Transform> _targetsByRacerId = new();
        private readonly IDisposable _subscription;
        private int _targetIndex = -1;
        private CinemachineCamera _orbitCamera;
        private OrbitCameraView _orbit;
        private CinemachineCamera _packCamera;
        private PackCameraView _pack;
        private Transform _orbitTarget;
        private bool _finalStretch;

        public Systems_CameraDirector(RaceModel model, CameraRigView rig, ISubscriber<LeadChangedMessage> leadChanged)
        {
            _model = model;
            _rig = rig;
            _subscription = leadChanged.Subscribe(OnLeadChanged);
            ShowOverview();
        }

        /// <summary>
        /// Final-stretch watcher: when the leading still-racing racer crosses the
        /// threshold, hand the shot to the orbit camera and keep it aimed at
        /// whoever is in front, so a last-second pass stays on screen.
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
            float fraction = front.Progress / Mathf.Max(1f, _model.TrackLengthMeters);
            if (!_finalStretch && fraction < FINAL_STRETCH_FRACTION)
            {
                return;
            }
            // Sticky until the next grid: flipping back to the pack shot when the
            // front runner finishes would cut away from the podium chase.
            _finalStretch = true;
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
            _finalStretch = false;
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

        public void NextTarget() => CycleTarget(1);

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

        public void Dispose() => _subscription?.Dispose();

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
        }
    }
}
