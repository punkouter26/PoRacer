using UnityEngine;

namespace IsaacBiped2
{
    /// <summary>
    /// Records how far one biped gets before it falls, so transfer quality can be judged from a
    /// distribution rather than a single run.
    ///
    /// Every Unity measurement in this project until now was n=1, and single draws have already
    /// produced misleading readings (the same policy measured 1.83 m once and 2.34 m another time).
    /// Spawn a batch of these with <c>IsaacBiped2BatchTest</c> and compare medians.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class IsaacBiped2DistanceProbe : MonoBehaviour
    {
        [Tooltip("dot(root.up, world.up) below this counts as fallen.")]
        [SerializeField] private float _fallenDot = 0.4f;

        private ArticulationBody _root;
        private Vector3 _origin;
        private bool _started;

        /// <summary>Furthest planar distance from the spawn point reached before falling.</summary>
        public float MaxDistance { get; private set; }

        /// <summary>Seconds of upright walking before the fall (or total elapsed if still up).</summary>
        public float TimeUpright { get; private set; }

        public bool Fallen { get; private set; }

        /// <summary>Lateral component of the lean at the moment of falling (|up.x|).</summary>
        public float FallLateral { get; private set; }

        /// <summary>Fore-aft component of the lean at the moment of falling (|up.z|).</summary>
        public float FallForeAft { get; private set; }

        private void Start()
        {
            _root = GetComponent<ArticulationBody>();
            _origin = _root.transform.position;
        }

        private void FixedUpdate()
        {
            if (_root == null || Fallen)
            {
                return;
            }
            // The agent pins the root until it finds ground; distance should only count from release.
            if (!_started)
            {
                if (_root.immovable)
                {
                    _origin = _root.transform.position;
                    return;
                }
                _started = true;
            }
            TimeUpright += Time.fixedDeltaTime;
            Vector3 delta = _root.transform.position - _origin;
            delta.y = 0f;
            float distance = delta.magnitude;
            if (distance > MaxDistance)
            {
                MaxDistance = distance;
            }
            Vector3 up = _root.transform.up;
            if (up.y < _fallenDot)
            {
                Fallen = true;
                FallLateral = Mathf.Abs(up.x);
                FallForeAft = Mathf.Abs(up.z);
            }
        }
    }
}
