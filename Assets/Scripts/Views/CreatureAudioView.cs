using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Per-creature 3D movement sound: a looping synthesized scuttle whose
    /// volume follows the creature's actual speed. Fully spatial — pan and
    /// attenuation come from the AudioListener on the camera. The clip is
    /// shared by all creatures; per-instance pitch offsets avoid phasing.
    /// </summary>
    public sealed class CreatureAudioView : MonoBehaviour
    {
        private const float FULL_VOLUME_SPEED = 2.5f;
        private const float MAX_VOLUME = 0.45f;

        private static AudioClip SharedScuttle;

        private AudioSource _source;
        private Transform _transform;
        private Vector3 _lastPosition;

        private void Awake()
        {
            _transform = transform;
            _lastPosition = _transform.position;
            _source = gameObject.AddComponent<AudioSource>();
            _source.clip = GetSharedScuttle();
            _source.loop = true;
            _source.playOnAwake = false;
            _source.spatialBlend = 1f;
            _source.dopplerLevel = 0f;
            _source.minDistance = 2f;
            _source.maxDistance = 40f;
            int entityId = GetEntityId().GetHashCode();
            _source.volume = 0f;
            _source.pitch = 0.85f + (entityId & 15) * 0.02f;
        }

        private void OnEnable()
        {
            if (_source != null && _source.clip != null)
            {
                _source.time = (GetEntityId().GetHashCode() & 7) * 0.11f;
                _source.Play();
            }
        }

        private void OnDisable()
        {
            if (_source != null)
            {
                _source.Stop();
            }
        }

        private void Update()
        {
            Vector3 position = _transform.position;
            float speed = (position - _lastPosition).magnitude / Mathf.Max(Time.deltaTime, 0.0001f);
            _lastPosition = position;
            float target = Mathf.Clamp01(speed / FULL_VOLUME_SPEED) * MAX_VOLUME;
            _source.volume = Mathf.MoveTowards(_source.volume, target, Time.deltaTime * 1.5f);
        }

        private static AudioClip GetSharedScuttle()
        {
            if (SharedScuttle != null)
            {
                return SharedScuttle;
            }
            const int sampleRate = 44100;
            const float seconds = 2f;
            int samples = (int)(sampleRate * seconds);
            var data = new float[samples];
            var rng = new System.Random(12345);
            float brown = 0f;
            for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
            {
                float white = (float)(rng.NextDouble() * 2.0 - 1.0);
                brown = Mathf.Clamp(brown + white * 0.12f, -1f, 1f) * 0.98f;
                float t = (float)sampleIndex / sampleRate;
                float patter = 0.55f + 0.45f * Mathf.Sin(2f * Mathf.PI * 7f * t);
                data[sampleIndex] = brown * patter * 0.5f;
            }
            var clip = AudioClip.Create("Scuttle", samples, 1, sampleRate, false);
            clip.SetData(data, 0);
            SharedScuttle = clip;
            return clip;
        }
    }
}
