using PoRacer.CreatureRace;
using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// Physics adapter for a worm on PhysX (the Isaac-trained worm, or any lane set to PhysX):
    /// reads the ArticulationBody tree every physics step, converts it into the MuJoCo
    /// convention the policy was trained in (WORM_SPEC "Unity mapping", CreatureFrames' SPEC
    /// map), hands it to the racer's <see cref="CreaturePilot"/>, and writes the targets into
    /// the drives. No decisions here.
    ///
    /// FixedUpdate runs once per PhysX step, before the step, reading the pose the last step
    /// left - the same moment a MuJoCo view reads its state, so every pilot counts the same
    /// physics steps.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class PhysxWormView : MonoBehaviour
    {
        private float[] _jointPositions;
        private float[] _jointVelocities;
        private float[] _targets;
        private Transform[] _bodies;
        private ArticulationBody _reference;
        private Transform _referenceTransform;
        private Transform _leadTransform;
        private Vector3 _leadOffset;
        private ArticulationBody[] _joints;
        private CreaturePilot _pilot;
        private Vector3 _goal;
        private float _passiveDamping;

        private void FixedUpdate()
        {
            if (_pilot == null)
            {
                return;
            }

            // B's axes: MuJoCo x = forward (Unity local +Z), y = left (Unity local -X),
            // z = up (Unity local +Y), each mapped into the SPEC world frame.
            Vector3 axisX = CreatureFrames.SpecFromUnityPolar(_referenceTransform.forward);
            Vector3 axisY = CreatureFrames.SpecFromUnityPolar(-_referenceTransform.right);
            Vector3 axisZ = CreatureFrames.SpecFromUnityPolar(_referenceTransform.up);
            // linearVelocity is the centre-of-mass velocity; the COM is the segment origin.
            Vector3 linear = CreatureFrames.SpecFromUnityPolar(_reference.linearVelocity);
            Vector3 angular = CreatureFrames.SpecFromUnityAxial(_reference.angularVelocity);

            for (int actionIndex = 0; actionIndex < _joints.Length; actionIndex++)
            {
                ArticulationBody joint = _joints[actionIndex];
                // Radians, and positive = positive MuJoCo qpos by the axial axis map.
                _jointPositions[actionIndex] = joint.jointPosition[0];
                _jointVelocities[actionIndex] = joint.jointVelocity[0];
            }

            var body = new CreatureBodyState(axisX, axisY, axisZ, linear, angular, _goal);
            bool targetsChanged = _pilot.Step(body, _jointPositions, _jointVelocities, ReadProbe(), _targets);

            for (int actionIndex = 0; actionIndex < _joints.Length; actionIndex++)
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

        /// <param name="bodies">Every rig body, rig order (WormRig and the adapted CreatureRig agree).</param>
        /// <param name="joints">The hinged bodies in action order.</param>
        internal void Bind(ArticulationBody[] bodies, ArticulationBody[] joints, CreaturePilot pilot,
                           Vector3 laneForwardUnity, CreatureLayout layout, float passiveDamping)
        {
            _passiveDamping = passiveDamping;
            _bodies = new Transform[bodies.Length];
            for (int bodyIndex = 0; bodyIndex < bodies.Length; bodyIndex++)
            {
                _bodies[bodyIndex] = bodies[bodyIndex].transform;
            }
            _reference = bodies[layout.ReferenceBody];
            _referenceTransform = _reference.transform;
            _leadTransform = _bodies[layout.LeadBody];
            _leadOffset = layout.LeadOffset;
            _joints = joints;
            _pilot = pilot;
            _jointPositions = new float[joints.Length];
            _jointVelocities = new float[joints.Length];
            _targets = new float[joints.Length];
            // The race direction in the SPEC MuJoCo frame: +x for a lane along Unity +Z,
            // exactly the goal the Isaac task trained with.
            _goal = CreatureFrames.SpecFromUnityPolar(laneForwardUnity);
        }

        /// <summary>Race and self-test telemetry in Unity space; probe offsets in MuJoCo convention.</summary>
        private CreatureProbe ReadProbe()
        {
            Transform bodyA = _bodies[_pilot.ProbeBodyA];
            Vector3 offset = PointOnBody(_bodies[_pilot.ProbeBodyB], _pilot.ProbePoint) - bodyA.position;
            var probeOffset = new Vector3(Vector3.Dot(offset, bodyA.forward),
                                          Vector3.Dot(offset, -bodyA.right),
                                          Vector3.Dot(offset, bodyA.up));
            return new CreatureProbe(PointOnBody(_leadTransform, _leadOffset), _referenceTransform.position.y,
                                     _referenceTransform.up.y, probeOffset);
        }

        /// <summary>A MuJoCo body-frame point on a PhysX body: x forward (+Z), y left (-X), z up (+Y).</summary>
        private static Vector3 PointOnBody(Transform body, Vector3 local)
        {
            return body.position + body.forward * local.x + -body.right * local.y + body.up * local.z;
        }
    }
}
