using UnityEngine;

using PoRacer.IsaacPorts;

namespace Boy
{
    /// <summary>
    /// Fallback target that reproduces the training command: a point on a ring around the
    /// creature, resampled when it gets within the reach radius or when the timer runs
    /// out. Used only when the scene supplies neither a Transform target nor another
    /// <see cref="ITargetProvider"/>. Defaults come from the rig asset's chase block.
    /// </summary>
    [DisallowMultipleComponent]
    public class BoyTargetSampler : MonoBehaviour, ITargetProvider
    {
        [Tooltip("Ring radius range [m]. Training used [3, 10].")]
        public Vector2 radiusRange = new Vector2(3f, 10f);

        [Tooltip("Resample when the creature is closer than this [m].")]
        public float reachRadius = 0.5f;

        [Tooltip("Resample after this many seconds regardless [s]. Training used [8, 12].")]
        public Vector2 resampleSeconds = new Vector2(8f, 12f);

        [Tooltip("Deterministic sampling so PlayMode runs repeat exactly.")]
        public int seed = 12345;

        [Tooltip("What the ring is centred on and measured from. Defaults to the articulation root.")]
        public Transform origin;

        Vector3 _current;
        float _nextResample;
        System.Random _rng;
        bool _started;

        /// <summary>How many targets have been reached since Start.</summary>
        public int Reached { get; private set; }

        public void ConfigureFrom(BoyRigAsset rig)
        {
            if (rig == null) return;
            radiusRange = new Vector2(rig.chase.targetRadiusMin, rig.chase.targetRadiusMax);
            reachRadius = rig.chase.reachRadius;
            resampleSeconds = new Vector2(rig.chase.resampleSecondsMin, rig.chase.resampleSecondsMax);
        }

        Vector3 Here()
        {
            if (origin != null) return origin.position;
            var body = GetComponentInChildren<ArticulationBody>();
            return body != null ? body.transform.position : transform.position;
        }

        void Start()
        {
            _rng = new System.Random(seed);
            _started = true;
            Resample();
        }

        void Update()
        {
            if (!_started) return;
            Vector3 here = Here();
            Vector3 d = _current - here;
            d.y = 0f;
            if (d.magnitude < reachRadius)
            {
                Reached++;
                Resample();
            }
            else if (Time.time >= _nextResample)
            {
                Resample();
            }
        }

        void Resample()
        {
            if (_rng == null) _rng = new System.Random(seed);
            Vector3 here = Here();
            double a = _rng.NextDouble() * System.Math.PI * 2.0;
            float rr = Mathf.Lerp(radiusRange.x, radiusRange.y, (float)_rng.NextDouble());
            _current = here + new Vector3((float)System.Math.Sin(a), 0f, (float)System.Math.Cos(a)) * rr;
            _current.y = here.y;
            _nextResample = Time.time + Mathf.Lerp(resampleSeconds.x, resampleSeconds.y, (float)_rng.NextDouble());
        }

        public bool TryGetTarget(out Vector3 worldPosition)
        {
            if (!_started) Start();
            worldPosition = _current;
            return true;
        }

        void OnDrawGizmosSelected()
        {
            if (!Application.isPlaying) return;
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireSphere(_current, reachRadius);
            Gizmos.DrawLine(Here(), _current);
        }
    }
}
