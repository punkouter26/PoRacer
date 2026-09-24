using UnityEngine;
using VContainer;

namespace PoRacer.WormRace
{
    /// <summary>
    /// Side-and-above follow camera that keeps the pack in frame: it tracks the average nose
    /// position of the racing worms along the track and stays centred on the lanes. Reads the model
    /// in LateUpdate because a camera has to move every frame; it decides nothing.
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(Camera))]
    public sealed class WormRaceCameraView : MonoBehaviour
    {
        [Tooltip("Camera position relative to the followed point (x = to the side, y = up).")]
        [SerializeField] private Vector3 _offset = new(5.5f, 3.2f, -3.0f);
        [Tooltip("How far ahead of the followed point the camera aims, metres.")]
        [SerializeField] private float _lookAhead = 1.5f;
        [Tooltip("Higher follows more tightly.")]
        [SerializeField] private float _followSharpness = 3f;

        private WormRaceModel _model;
        private Transform _transform;

        [Inject]
        public void Construct(WormRaceModel model)
        {
            _model = model;
        }

        private void Awake()
        {
            _transform = transform;
        }

        private void LateUpdate()
        {
            if (_model == null)
            {
                return;
            }
            Vector3 focus = Focus();
            Vector3 desired = focus + _offset;
            float blend = 1f - Mathf.Exp(-_followSharpness * Time.deltaTime);
            _transform.position = Vector3.Lerp(_transform.position, desired, blend);
            Vector3 aim = focus + Vector3.forward * _lookAhead - _transform.position;
            if (aim.sqrMagnitude > 1e-6f)
            {
                _transform.rotation = Quaternion.LookRotation(aim, Vector3.up);
            }
        }

        private Vector3 Focus()
        {
            if (!_model.WormsSpawned)
            {
                return _model.StartFocus;
            }
            // Follow the worms that are actually racing: a NO BRAIN worm lies on the start
            // line and would drag the frame back. Everyone still in, if none has a brain.
            if (!TryAverageNose(true, out float noseZ) && !TryAverageNose(false, out noseZ))
            {
                return _model.StartFocus;
            }
            // Centred between the lanes, on the floor, at the pack's average nose.
            return new Vector3(_model.StartFocus.x, 0f, noseZ);
        }

        private bool TryAverageNose(bool brainedOnly, out float noseZ)
        {
            float noseSum = 0f;
            int counted = 0;
            for (int lane = 0; lane < _model.Racers.Count; lane++)
            {
                WormRacerModel racer = _model.Racers[lane];
                if (racer.Status == WormRacerStatus.Failed || (brainedOnly && !racer.BrainReady))
                {
                    continue;
                }
                noseSum += racer.Nose.z;
                counted++;
            }
            noseZ = counted > 0 ? noseSum / counted : 0f;
            return counted > 0;
        }
    }
}
