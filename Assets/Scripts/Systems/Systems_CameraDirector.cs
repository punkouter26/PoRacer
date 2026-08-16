using System;
using System.Collections.Generic;
using MessagePipe;
using PoRacer.Models;
using PoRacer.Views;
using UnityEngine;

namespace PoRacer.Systems
{
    /// <summary>
    /// Presentation referee: switches between the overview camera and a chase camera
    /// that follows one racer. InputView calls NextTarget/PrevTarget/Overview.
    /// Every lead change re-aims the chase camera at the new leader, so the shot
    /// follows the story of the race by default; manual cycling still works.
    /// Cinemachine 3.x: priority switching + Target.TrackingTarget reassignment.
    /// </summary>
    public sealed class Systems_CameraDirector : IDisposable
    {
        private const int ACTIVE_PRIORITY = 20;
        private const int INACTIVE_PRIORITY = 0;

        private readonly CameraRigView _rig;
        private readonly List<Transform> _targets = new();
        private readonly Dictionary<string, Transform> _targetsByRacerId = new();
        private readonly IDisposable _subscription;
        private int _targetIndex = -1;

        public Systems_CameraDirector(CameraRigView rig, ISubscriber<LeadChangedMessage> leadChanged)
        {
            _rig = rig;
            _subscription = leadChanged.Subscribe(OnLeadChanged);
            ShowOverview();
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
            ShowOverview();
        }

        /// <summary>Spawn registers each racer so lead changes can find its transform.</summary>
        public void RegisterRacer(string racerId, Transform target)
        {
            _targetsByRacerId[racerId] = target;
        }

        private void OnLeadChanged(LeadChangedMessage message)
        {
            if (!_targetsByRacerId.TryGetValue(message.RacerId, out Transform leader)
                || leader == null || !leader.gameObject.activeInHierarchy)
            {
                return;
            }
            _rig.ChaseCamera.Target.TrackingTarget = leader;
            _rig.ChaseCamera.Target.LookAtTarget = leader;
            _rig.ChaseCamera.Priority = ACTIVE_PRIORITY;
            _rig.OverviewCamera.Priority = INACTIVE_PRIORITY;
        }

        public void NextTarget() => CycleTarget(1);

        public void PrevTarget() => CycleTarget(-1);

        public void ShowOverview()
        {
            _rig.OverviewCamera.Priority = ACTIVE_PRIORITY;
            _rig.ChaseCamera.Priority = INACTIVE_PRIORITY;
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
            _rig.ChaseCamera.Target.TrackingTarget = target;
            _rig.ChaseCamera.Target.LookAtTarget = target;
            _rig.ChaseCamera.Priority = ACTIVE_PRIORITY;
            _rig.OverviewCamera.Priority = INACTIVE_PRIORITY;
        }
    }
}
