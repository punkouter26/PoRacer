using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// Physics adapter for the Isaac-trained worm on PhysX: reads the ArticulationBody
    /// chain every physics step, converts it into the MuJoCo convention the policy was
    /// trained in (WORM_SPEC "Unity mapping"), hands it to the worm's
    /// <see cref="WormPilot"/>, and writes the targets into the drives. No decisions here.
    ///
    /// FixedUpdate runs once per PhysX step, before the step, reading the pose the last
    /// step left - the same moment the MuJoCo view reads its state, so both pilots count
    /// the same physics steps.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class PhysxWormView : MonoBehaviour
    {
        private readonly float[] _jointPositions = new float[WormContract.ACTION_SIZE];
        private readonly float[] _jointVelocities = new float[WormContract.ACTION_SIZE];
        private readonly float[] _targets = new float[WormContract.ACTION_SIZE];

        private ArticulationBody _reference;
        private Transform _referenceTransform;
        private Transform _headTransform;
        private Transform _secondTransform;
        private ArticulationBody[] _joints;
        private WormPilot _pilot;
        private Vector3 _goal;
        private float _noseOffset;
        private float _passiveDamping;

        internal void Bind(ArticulationBody[] segments, ArticulationBody[] joints, WormPilot pilot,
                           Vector3 laneForwardUnity, float noseOffset, float passiveDamping)
        {
            _passiveDamping = passiveDamping;
            _reference = segments[WormContract.REFERENCE_SEGMENT];
            _referenceTransform = _reference.transform;
            _headTransform = segments[WormContract.HEAD_SEGMENT].transform;
            _secondTransform = segments[WormContract.SECOND_SEGMENT].transform;
            _joints = joints;
            _pilot = pilot;
            _noseOffset = noseOffset;
            // The race direction in the SPEC MuJoCo frame: +x for a lane along Unity +Z,
            // exactly the goal the Isaac task trained with.
            _goal = WormFrames.SpecFromUnityPolar(laneForwardUnity);
        }

        private void FixedUpdate()
        {
            if (_pilot == null)
            {
                return;
            }

            // B's axes: MuJoCo x = forward (Unity local +Z), y = left (Unity local -X),
            // z = up (Unity local +Y), each mapped into the SPEC world frame.
            Vector3 axisX = WormFrames.SpecFromUnityPolar(_referenceTransform.forward);
            Vector3 axisY = WormFrames.SpecFromUnityPolar(-_referenceTransform.right);
            Vector3 axisZ = WormFrames.SpecFromUnityPolar(_referenceTransform.up);
            // linearVelocity is the centre-of-mass velocity; the COM is the segment origin.
            Vector3 linear = WormFrames.SpecFromUnityPolar(_reference.linearVelocity);
            Vector3 angular = WormFrames.SpecFromUnityAxial(_reference.angularVelocity);

            for (int actionIndex = 0; actionIndex < WormContract.ACTION_SIZE; actionIndex++)
            {
                ArticulationBody joint = _joints[actionIndex];
                // Radians, and positive = positive MuJoCo qpos by the axial axis map.
                _jointPositions[actionIndex] = joint.jointPosition[0];
                _jointVelocities[actionIndex] = joint.jointVelocity[0];
            }

            Vector3 headPosition = _headTransform.position;
            Vector3 offset = _secondTransform.position - headPosition;
            var probe = new WormProbe(
                headPosition + _headTransform.forward * _noseOffset,
                _referenceTransform.position.y,
                Vector3.Dot(offset, _headTransform.right),
                Vector3.Dot(offset, _headTransform.up));

            var body = new WormBodyState(axisX, axisY, axisZ, linear, angular, _goal);
            bool targetsChanged = _pilot.Step(body, _jointPositions, _jointVelocities, probe, _targets);

            for (int actionIndex = 0; actionIndex < WormContract.ACTION_SIZE; actionIndex++)
            {
                ArticulationBody joint = _joints[actionIndex];
                if (targetsChanged)
                {
                    ArticulationDrive drive = joint.xDrive;
                    drive.target = _targets[actionIndex] * Mathf.Rad2Deg;
                    joint.xDrive = drive;
                }
                float jointVelocity = _jointVelocities[actionIndex];
                if (_passiveDamping > 0f && float.IsFinite(jointVelocity))
                {
                    // WORM_SPEC detail 16: passive -c*qdot outside the servo's force limit,
                    // re-applied every step from this step's joint velocity.
                    joint.jointForce = new ArticulationReducedSpace(-_passiveDamping * jointVelocity);
                }
            }
        }
    }
}
