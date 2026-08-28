using UnityEngine;

namespace IsaacH1
{
    /// <summary>
    /// Fallback target: samples points on a ring around a centre, the way Isaac's
    /// UniformVelocityCommand resamples a heading every resampling_time_range seconds.
    /// Used only when the scene supplies neither a Transform nor an ITargetProvider.
    ///
    /// The ring radius is deliberately larger than the distance the creature covers in
    /// one resample interval, so the commanded heading stays roughly constant between
    /// resamples - matching how the policy was trained.
    /// </summary>
    [DisallowMultipleComponent]
    public class IsaacH1RingTargetSampler : MonoBehaviour, ITargetProvider
    {
        [Tooltip("Ring centre. Defaults to this object's position at Start.")]
        public Transform center;

        [Tooltip("Isaac resamples the command every 10 s (resampling_time_range).")]
        public float resampleSeconds = 10f;

        public float radius = 12f;

        [Tooltip("Deterministic sampling so PlayMode runs repeat exactly.")]
        public int seed = 12345;

        Vector3 _center;
        Vector3 _current;
        float _nextResample;
        System.Random _rng;

        void Start()
        {
            _center = center != null ? center.position : transform.position;
            _rng = new System.Random(seed);
            Resample();
        }

        void Update()
        {
            if (Time.time >= _nextResample) Resample();
        }

        void Resample()
        {
            if (_rng == null) _rng = new System.Random(seed);
            double a = _rng.NextDouble() * System.Math.PI * 2.0;
            _current = _center + new Vector3((float)System.Math.Sin(a), 0f, (float)System.Math.Cos(a)) * radius;
            _nextResample = Time.time + resampleSeconds;
        }

        public bool TryGetTarget(out Vector3 worldPosition)
        {
            if (_rng == null) Start();
            worldPosition = _current;
            return true;
        }

        void OnDrawGizmosSelected()
        {
            Vector3 c = Application.isPlaying ? _center : (center != null ? center.position : transform.position);
            Gizmos.color = new Color(0.2f, 0.8f, 1f, 0.6f);
            const int n = 48;
            for (int i = 0; i < n; i++)
            {
                float a0 = i * Mathf.PI * 2f / n, a1 = (i + 1) * Mathf.PI * 2f / n;
                Gizmos.DrawLine(c + new Vector3(Mathf.Sin(a0), 0, Mathf.Cos(a0)) * radius,
                                c + new Vector3(Mathf.Sin(a1), 0, Mathf.Cos(a1)) * radius);
            }
            if (Application.isPlaying)
            {
                Gizmos.color = Color.yellow;
                Gizmos.DrawWireSphere(_current, 0.35f);
            }
        }
    }
}
