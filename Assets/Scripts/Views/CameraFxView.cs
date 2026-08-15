using MessagePipe;
using PoRacer.Models;
using Unity.Cinemachine;
using UnityEngine;
using VContainer;

namespace PoRacer.Views
{
    /// <summary>
    /// Camera juice: impulse shake on race start and on a first-place finish.
    /// Lives on the camera rig; adds impulse listeners to every Cinemachine
    /// camera underneath it at runtime.
    /// </summary>
    public sealed class CameraFxView : MonoBehaviour
    {
        private const float START_SHAKE = 0.25f;
        private const float WIN_SHAKE = 0.6f;

        private CinemachineImpulseSource _impulse;
        private System.IDisposable _subscriptions;

        [Inject]
        public void Construct(
            ISubscriber<RaceStartedMessage> raceStarted,
            ISubscriber<RacerFinishedMessage> racerFinished)
        {
            var bag = DisposableBag.CreateBuilder();
            raceStarted.Subscribe(OnRaceStarted).AddTo(bag);
            racerFinished.Subscribe(OnRacerFinished).AddTo(bag);
            _subscriptions = bag.Build();
        }

        private void Awake()
        {
            _impulse = gameObject.AddComponent<CinemachineImpulseSource>();
            _impulse.ImpulseDefinition.ImpulseDuration = 0.35f;
            CinemachineCamera[] cameras = GetComponentsInChildren<CinemachineCamera>(true);
            for (int cameraIndex = 0; cameraIndex < cameras.Length; cameraIndex++)
            {
                cameras[cameraIndex].gameObject.AddComponent<CinemachineImpulseListener>();
            }
        }

        private void OnDestroy() => _subscriptions?.Dispose();

        private void OnRaceStarted(RaceStartedMessage message) => _impulse.GenerateImpulse(START_SHAKE);

        private void OnRacerFinished(RacerFinishedMessage message)
        {
            if (message.Place == 1)
            {
                _impulse.GenerateImpulse(WIN_SHAKE);
            }
        }
    }
}
