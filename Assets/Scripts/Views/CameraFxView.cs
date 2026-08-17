using MessagePipe;
using PoRacer.Models;
using Unity.Cinemachine;
using UnityEngine;
using VContainer;

namespace PoRacer.Views
{
    /// <summary>
    /// Camera juice: impulse shake on race start, lead changes, and a
    /// first-place finish, plus a short slow-motion beat when the winner
    /// crosses the line. Lives on the camera rig; adds impulse listeners to
    /// every Cinemachine camera underneath it at runtime.
    /// </summary>
    public sealed class CameraFxView : MonoBehaviour
    {
        private const float START_SHAKE = 0.25f;
        private const float LEAD_SHAKE = 0.12f;
        private const float WIN_SHAKE = 0.6f;
        // Winner slow-mo: drop to this timescale, hold, then ease back to 1.
        private const float SLOWMO_SCALE = 0.35f;
        private const float SLOWMO_SECONDS = 0.9f;
        private const float SLOWMO_RECOVERY_PER_SECOND = 1.6f;
        // Tight clip range: the scene authored 0.1-5000, which wrecks depth
        // precision and z-fights the ground decals at distance.
        private const float NEAR_CLIP = 0.3f;
        private const float FAR_CLIP = 400f;

        private CinemachineImpulseSource _impulse;
        private System.IDisposable _subscriptions;
        private float _slowMoUntil;

        [Inject]
        public void Construct(
            ISubscriber<RaceStartedMessage> raceStarted,
            ISubscriber<LeadChangedMessage> leadChanged,
            ISubscriber<RacerFinishedMessage> racerFinished)
        {
            var bag = DisposableBag.CreateBuilder();
            raceStarted.Subscribe(OnRaceStarted).AddTo(bag);
            leadChanged.Subscribe(OnLeadChanged).AddTo(bag);
            racerFinished.Subscribe(OnRacerFinished).AddTo(bag);
            _subscriptions = bag.Build();
        }

        private void Awake()
        {
            _impulse = gameObject.AddComponent<CinemachineImpulseSource>();
            _impulse.ImpulseDefinition.ImpulseDuration = 0.35f;
            Camera mainCamera = Camera.main;
            if (mainCamera != null)
            {
                mainCamera.nearClipPlane = NEAR_CLIP;
                mainCamera.farClipPlane = FAR_CLIP;
            }
            CinemachineCamera[] cameras = GetComponentsInChildren<CinemachineCamera>(true);
            for (int cameraIndex = 0; cameraIndex < cameras.Length; cameraIndex++)
            {
                cameras[cameraIndex].gameObject.AddComponent<CinemachineImpulseListener>();
                LensSettings lens = cameras[cameraIndex].Lens;
                lens.NearClipPlane = NEAR_CLIP;
                lens.FarClipPlane = FAR_CLIP;
                cameras[cameraIndex].Lens = lens;
            }
        }

        private void Update()
        {
            if (Time.unscaledTime < _slowMoUntil)
            {
                Time.timeScale = SLOWMO_SCALE;
            }
            else if (Time.timeScale < 1f)
            {
                Time.timeScale = Mathf.MoveTowards(
                    Time.timeScale, 1f, Time.unscaledDeltaTime * SLOWMO_RECOVERY_PER_SECOND);
            }
        }

        private void OnDisable()
        {
            // Disabled mid-slow-mo, Update stops running: the recovery ramp must
            // not leave the whole game stuck below full speed.
            _slowMoUntil = 0f;
            Time.timeScale = 1f;
        }

        private void OnDestroy()
        {
            _subscriptions?.Dispose();
            Time.timeScale = 1f;
        }

        private void OnRaceStarted(RaceStartedMessage message)
        {
            _impulse.GenerateImpulse(START_SHAKE);
            CrowdMood.Excite();
        }

        private void OnLeadChanged(LeadChangedMessage message)
        {
            _impulse.GenerateImpulse(LEAD_SHAKE);
            CrowdMood.Excite();
        }

        private void OnRacerFinished(RacerFinishedMessage message)
        {
            if (message.Place == 1)
            {
                _impulse.GenerateImpulse(WIN_SHAKE);
                _slowMoUntil = Time.unscaledTime + SLOWMO_SECONDS;
            }
            CrowdMood.Excite();
        }
    }
}
