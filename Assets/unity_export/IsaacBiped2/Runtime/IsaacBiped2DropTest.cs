using System.Text;
using UnityEngine;

namespace IsaacBiped2
{
    /// <summary>
    /// Records torso height every physics step through the landing transient, to find where the rig
    /// gains energy. The open-loop replay showed Unity reaching 0.720 m from a 0.68 m spawn against a
    /// zero-restitution material, which should be impossible under gravity alone.
    ///
    /// Set <see cref="_holdFirst"/> to compare the two release paths: a free spawn versus the
    /// pinned-then-unpinned path the agent uses (immovable = true, then false). If only the held path
    /// overshoots, the hold is injecting the energy, not the contact.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class IsaacBiped2DropTest : MonoBehaviour
    {
        private const int ACT = 10;
        private const int SAMPLES = 400;

        [Tooltip("Pin the root for a settle period and then release, the way the agent does.")]
        [SerializeField] private bool _holdFirst = true;
        [SerializeField] private int _holdSteps = 200;

        private static readonly string[] Links =
        {
            "L_hip_yaw_link", "L_hip_roll_link", "L_thigh", "L_shank", "L_foot",
            "R_hip_yaw_link", "R_hip_roll_link", "R_thigh", "R_shank", "R_foot"
        };

        private static readonly float[] DefaultPose =
        {
            0f, 0f, -0.25f, 0.50f, -0.25f,
            0f, 0f, -0.25f, 0.50f, -0.25f
        };

        private ArticulationBody _root;
        private int _step;
        private bool _released;
        private int _sample;

        public float[] Height { get; } = new float[SAMPLES];
        public float[] VelY { get; } = new float[SAMPLES];
        public bool Finished { get; private set; }
        public float PeakHeight { get; private set; }
        public float SpawnHeight { get; private set; }

        private void Start()
        {
            _root = GetComponent<ArticulationBody>();
            SpawnHeight = _root.transform.position.y;
            PeakHeight = SpawnHeight;
            ArticulationBody[] all = GetComponentsInChildren<ArticulationBody>(true);
            for (int index = 0; index < ACT; index++)
            {
                for (int bodyIndex = 0; bodyIndex < all.Length; bodyIndex++)
                {
                    if (all[bodyIndex].name != Links[index])
                    {
                        continue;
                    }
                    ArticulationDrive drive = all[bodyIndex].xDrive;
                    drive.target = DefaultPose[index] * Mathf.Rad2Deg;
                    all[bodyIndex].xDrive = drive;
                    break;
                }
            }
            if (_holdFirst)
            {
                _root.immovable = true;
            }
            else
            {
                _released = true;
            }
        }

        private void FixedUpdate()
        {
            if (Finished)
            {
                return;
            }
            _step++;
            if (!_released)
            {
                if (_step >= _holdSteps)
                {
                    _root.immovable = false;
                    _released = true;
                    _step = 0;
                }
                return;
            }
            if (_sample >= SAMPLES)
            {
                Finished = true;
                Debug.Log($"[IsaacBiped2DropTest] hold={_holdFirst} spawn={SpawnHeight:F3} " +
                          $"peak={PeakHeight:F3} overshoot={PeakHeight - SpawnHeight:+0.000;-0.000} m");
                return;
            }
            float y = _root.transform.position.y;
            Height[_sample] = y;
            VelY[_sample] = _root.linearVelocity.y;
            if (y > PeakHeight)
            {
                PeakHeight = y;
            }
            _sample++;
        }

        /// <summary>Compact trace for the diagnostic: every Nth sample.</summary>
        public string Trace(int stride)
        {
            var builder = new StringBuilder();
            for (int index = 0; index < SAMPLES && index < _sample; index += stride)
            {
                builder.Append((index * Time.fixedDeltaTime).ToString("F3")).Append("s:")
                    .Append(Height[index].ToString("F3")).Append("/")
                    .Append(VelY[index].ToString("+0.00;-0.00")).Append(" ");
            }
            return builder.ToString();
        }
    }
}
