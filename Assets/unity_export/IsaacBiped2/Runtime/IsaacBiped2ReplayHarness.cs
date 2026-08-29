using System;
using System.Text;
using UnityEngine;

namespace IsaacBiped2
{
    /// <summary>
    /// Open-loop dynamics comparison against Isaac. Replays the exact action sequence recorded in
    /// <c>isaac_reference.json</c> with no policy in the loop, so any divergence between Unity's
    /// joint trajectory and Isaac's is dynamics, not control.
    ///
    /// This is the test that separates "the rig transfers and the policy is fragile" from "the rig
    /// itself does not reproduce Isaac's dynamics". Attach to a built IsaacBiped2 prefab instance
    /// with <see cref="IsaacBiped2Agent"/> removed.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class IsaacBiped2ReplayHarness : MonoBehaviour
    {
        [Serializable]
        private sealed class Step
        {
            public float[] action;
            public float[] joint_pos;
            public float[] root_pos_w;
        }

        [Serializable]
        private sealed class Wrapper
        {
            public Step[] steps;
        }

        private const int ACT = 10;

        [SerializeField] private TextAsset _reference;
        [Tooltip("Physics steps per recorded action. Isaac ran 4 at 0.005 s.")]
        [SerializeField] private int _decimation = 4;
        [Tooltip("Physics steps spent servoing to the reference start pose before replay begins.")]
        [SerializeField] private int _settleSteps = 400;
        [SerializeField] private float _actionScale = 0.5f;

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

        private Step[] _steps;
        private ArticulationBody _root;
        private readonly ArticulationBody[] _joints = new ArticulationBody[ACT];
        private int _physicsStep;
        private int _cursor;
        private bool _replaying;

        /// <summary>Mean |dq| over the 10 joints, per replayed step. Read by the diagnostic.</summary>
        public float[] JointError { get; private set; }

        /// <summary>Unity torso height minus Isaac's, per replayed step.</summary>
        public float[] HeightError { get; private set; }

        /// <summary>Unity torso height, per replayed step.</summary>
        public float[] UnityHeight { get; private set; }

        /// <summary>Signed per-joint error (unity - isaac) at each replayed step, [step][joint].</summary>
        public float[][] PerJointError { get; private set; }

        /// <summary>Unity joint angle at each replayed step, [step][joint].</summary>
        public float[][] UnityJoint { get; private set; }

        /// <summary>Isaac joint angle at each replayed step, [step][joint].</summary>
        public float[][] IsaacJoint { get; private set; }

        /// <summary>How many steps each joint spent within 0.02 rad of a drive limit.</summary>
        public int[] LimitHits { get; private set; }

        public bool Finished { get; private set; }

        private void Start()
        {
            _root = GetComponent<ArticulationBody>();
            ArticulationBody[] all = GetComponentsInChildren<ArticulationBody>(true);
            for (int index = 0; index < ACT; index++)
            {
                for (int bodyIndex = 0; bodyIndex < all.Length; bodyIndex++)
                {
                    if (all[bodyIndex].name == Links[index])
                    {
                        _joints[index] = all[bodyIndex];
                        break;
                    }
                }
            }
            // JsonUtility cannot parse a bare top-level array, so wrap it in an object.
            _steps = JsonUtility.FromJson<Wrapper>("{\"steps\":" + _reference.text + "}").steps;
            JointError = new float[_steps.Length];
            HeightError = new float[_steps.Length];
            UnityHeight = new float[_steps.Length];
            PerJointError = new float[_steps.Length][];
            UnityJoint = new float[_steps.Length][];
            IsaacJoint = new float[_steps.Length][];
            LimitHits = new int[ACT];
            // Servo to Isaac's recorded starting pose (it randomises the reset pose slightly, and an
            // unstable rig amplifies any initial offset, so matching it matters).
            SetTargets(_steps[0].joint_pos);
            _root.immovable = true;
        }

        private void FixedUpdate()
        {
            if (Finished)
            {
                return;
            }
            _physicsStep++;
            if (!_replaying)
            {
                if (_physicsStep == _settleSteps)
                {
                    _root.immovable = false;
                    _replaying = true;
                    _physicsStep = 0;
                }
                return;
            }
            if (_physicsStep % _decimation != 0)
            {
                return;
            }
            if (_cursor >= _steps.Length)
            {
                Finished = true;
                Report();
                return;
            }
            Step step = _steps[_cursor];
            // Same mapping the env uses: target = default_q + action_scale * clamp(action, -1, 1).
            var targets = new float[ACT];
            for (int index = 0; index < ACT; index++)
            {
                targets[index] = DefaultPose[index] + _actionScale * Mathf.Clamp(step.action[index], -1f, 1f);
            }
            SetTargets(targets);

            float sum = 0f;
            PerJointError[_cursor] = new float[ACT];
            UnityJoint[_cursor] = new float[ACT];
            IsaacJoint[_cursor] = new float[ACT];
            for (int index = 0; index < ACT; index++)
            {
                float q = _joints[index].jointPosition[0];
                sum += Mathf.Abs(q - step.joint_pos[index]);
                PerJointError[_cursor][index] = q - step.joint_pos[index];
                UnityJoint[_cursor][index] = q;
                IsaacJoint[_cursor][index] = step.joint_pos[index];
                // Sitting on a limit means the drive is being overruled by the joint stop, which
                // produces a very different force profile from the same angle reached freely.
                ArticulationDrive drive = _joints[index].xDrive;
                float deg = q * Mathf.Rad2Deg;
                if (deg - drive.lowerLimit < 1.2f || drive.upperLimit - deg < 1.2f)
                {
                    LimitHits[index]++;
                }
            }
            JointError[_cursor] = sum / ACT;
            UnityHeight[_cursor] = _root.transform.position.y;
            HeightError[_cursor] = _root.transform.position.y - step.root_pos_w[2];
            _cursor++;
        }

        private void SetTargets(float[] radians)
        {
            for (int index = 0; index < ACT; index++)
            {
                if (_joints[index] == null)
                {
                    continue;
                }
                ArticulationDrive drive = _joints[index].xDrive;
                drive.target = radians[index] * Mathf.Rad2Deg;
                _joints[index].xDrive = drive;
            }
        }

        private void Report()
        {
            var builder = new StringBuilder();
            builder.Append("[IsaacBiped2Replay] open-loop vs Isaac over ").Append(_cursor).Append(" steps:\n");
            int[] marks = { 25, 50, 100, 150, 200, 249 };
            for (int index = 0; index < marks.Length; index++)
            {
                int k = marks[index];
                if (k >= _cursor)
                {
                    continue;
                }
                builder.Append("  t=").Append((k * 0.02f).ToString("F2")).Append("s  mean|dq|=")
                    .Append(JointError[k].ToString("F4")).Append(" rad  unity_z=")
                    .Append(UnityHeight[k].ToString("F3")).Append("  dz=")
                    .Append(HeightError[k].ToString("+0.000;-0.000")).Append("\n");
            }
            Debug.Log(builder.ToString());
        }
    }
}
