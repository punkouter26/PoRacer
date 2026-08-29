using UnityEngine;

using PoRacer.IsaacPorts;

namespace MujocoBiped
{
    /// <summary>
    /// Fallback target, used only when the scene supplies neither a Transform nor an
    /// <see cref="ITargetProvider"/>. It reproduces env.py's <c>_sample_target</c>: a new
    /// goal 3-6 m away, within +/-138 degrees of the CURRENT heading, respawned whenever
    /// the creature gets within the 0.6 m reach radius.
    ///
    /// Matching the training distribution matters more than it looks. Reaching more than
    /// one goal per episode requires turning, and the policy only ever saw goals inside
    /// that cone at those distances - a goal 30 m away and dead ahead is out of
    /// distribution in a way that shows up as a worse gait, not as an error.
    /// </summary>
    [DisallowMultipleComponent]
    [AddComponentMenu("MujocoBiped/MujocoBiped Target Sampler")]
    public class MujocoBipedTargetSampler : MonoBehaviour, ITargetProvider
    {
        [Tooltip("Deterministic sampling, so a PlayMode run repeats exactly.")]
        public int seed = 12345;

        [Tooltip("env.py target_distance_range_m.")]
        public Vector2 distanceRange = new Vector2(3f, 6f);

        [Tooltip("env.py target_angle_range_rad: +/-2.4 rad = +/-137.5 degrees of heading.")]
        public float angleRangeRad = 2.4f;

        [Tooltip("env.py reach_radius_m. Inside this the goal is 'reached' and respawns.")]
        public float reachRadius = 0.6f;

        [Tooltip("Goal marker height, matching the MJCF mocap body at z = 0.02.")]
        public float targetHeight = 0.02f;

        Vector3 _current;
        bool _has;
        int _reached;
        System.Random _rng;

        /// <summary>How many goals have been reached - the metric MuJoCo's eval reports.</summary>
        public int TargetsReached => _reached;

        void Start() => Resample();

        void Update()
        {
            if (!_has) { Resample(); return; }
            Vector3 d = transform.position - _current;
            d.y = 0f;
            if (d.magnitude <= reachRadius)
            {
                _reached++;
                Resample();
            }
        }

        /// <summary>Places a new goal relative to where the creature is and faces now.</summary>
        public void Resample()
        {
            if (_rng == null) _rng = new System.Random(seed);

            float heading = MujocoBipedFrameMap.HeadingRad(transform.rotation);
            float distance = Mathf.Lerp(distanceRange.x, distanceRange.y, (float)_rng.NextDouble());
            float angle = heading + ((float)_rng.NextDouble() * 2f - 1f) * angleRangeRad;

            // env.py samples in MuJoCo's XY plane: xy = origin + d * (cos a, sin a).
            // MuJoCo +X is Unity +Z and MuJoCo +Y is Unity -X.
            Vector3 here = transform.position;
            _current = new Vector3(here.x - distance * Mathf.Sin(angle),
                                   targetHeight,
                                   here.z + distance * Mathf.Cos(angle));
            _has = true;
        }

        public bool TryGetTarget(out Vector3 worldPosition)
        {
            if (!_has) Resample();
            worldPosition = _current;
            return true;
        }

        void OnDrawGizmosSelected()
        {
            if (!_has) return;
            Gizmos.color = new Color(0.9f, 0.2f, 0.2f, 0.8f);
            Gizmos.DrawWireSphere(_current, reachRadius);
            Gizmos.DrawLine(_current, _current + Vector3.up * 0.9f);
        }
    }
}
