using Mujoco;
using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// Physics adapter for a MuJoCo-plug-in racer of any creature: reads MuJoCo's own state
    /// inside the plug-in's control callback, hands it to the racer's
    /// <see cref="CreaturePilot"/>, and writes the returned targets into the position servos.
    /// No decisions are made here.
    ///
    /// Everything is read straight out of mjData, in MuJoCo's frame, not off the Unity
    /// transforms the plug-in mirrors, so no frame conversion can drift between what MuJoCo
    /// simulates and what the policy sees (the reference body's velocity comes from
    /// mj_objectVelocity, the quantity the trainers compute).
    ///
    /// TIMING. MjScene.StepScene runs mj_step1, then this callback, then mj_step2. So the
    /// state read here is the state the coming step integrates from, and ctrl written here
    /// takes effect in THIS step - unlike MjActuator.Control, which the plug-in copies into
    /// ctrl only after the step (a one-step lag). Both are written: ctrl directly for zero
    /// lag, Control so the plug-in's post-step copy writes the same value back.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class MujocoCreatureView : MonoBehaviour
    {
        private const int VELOCITY_SIZE = 6;
        private const int MATRIX_SIZE = 9;
        private const int VECTOR_SIZE = 3;

        private readonly double[] _velocity = new double[VELOCITY_SIZE];

        private float[] _jointPositions;
        private float[] _jointVelocities;
        private float[] _targets;
        private MjBody[] _bodies;
        private MjHingeJoint[] _joints;
        private MjActuator[] _actuators;
        private CreaturePilot _pilot;
        private int _referenceBody;
        private int _leadBody;
        private Vector3 _leadOffset;
        private Vector3 _goal;
        private bool _subscribed;

        private void OnEnable()
        {
            if (_pilot != null)
            {
                Subscribe();
            }
        }

        private void OnDisable()
        {
            Unsubscribe();
        }

        /// <param name="joints">Hinges in action order.</param>
        /// <param name="actuators">Servos in action order.</param>
        public void Bind(MjBody[] bodies, MjHingeJoint[] joints, MjActuator[] actuators, CreaturePilot pilot,
                         Vector3 laneForwardUnity, CreatureLayout layout)
        {
            _bodies = bodies;
            _joints = joints;
            _actuators = actuators;
            _pilot = pilot;
            _referenceBody = layout.ReferenceBody;
            _leadBody = layout.LeadBody;
            _leadOffset = layout.LeadOffset;
            _jointPositions = new float[layout.ActionSize];
            _jointVelocities = new float[layout.ActionSize];
            _targets = new float[layout.ActionSize];
            // The race direction in the plug-in's MuJoCo world frame.
            _goal = CreatureFrames.PluginFromUnity(laneForwardUnity);
            Subscribe();
        }

        private void Subscribe()
        {
            // InstanceExists, never Instance: the getter CREATES an MjScene when there is
            // none, and a second one is exactly the singleton error to avoid.
            if (_subscribed || !MjScene.InstanceExists)
            {
                return;
            }
            MjScene.Instance.ctrlCallback += OnControl;
            _subscribed = true;
        }

        private void Unsubscribe()
        {
            if (!_subscribed)
            {
                return;
            }
            if (MjScene.InstanceExists)
            {
                MjScene.Instance.ctrlCallback -= OnControl;
            }
            _subscribed = false;
        }

        private unsafe void OnControl(object sender, MjStepArgs args)
        {
            MujocoLib.mjModel_* model = args.model;
            MujocoLib.mjData_* data = args.data;
            if (model == null || data == null || _pilot == null)
            {
                return;
            }

            int referenceId = _bodies[_referenceBody].MujocoId;
            double* referenceMatrix = data->xmat + MATRIX_SIZE * referenceId;
            // xmat is row-major body-to-world: B's axis c is column c.
            var axisX = new Vector3((float)referenceMatrix[0], (float)referenceMatrix[3], (float)referenceMatrix[6]);
            var axisY = new Vector3((float)referenceMatrix[1], (float)referenceMatrix[4], (float)referenceMatrix[7]);
            var axisZ = new Vector3((float)referenceMatrix[2], (float)referenceMatrix[5], (float)referenceMatrix[8]);

            Vector3 angular;
            Vector3 linear;
            fixed (double* velocity = _velocity)
            {
                // XBODY: at the body frame origin, world orientation (flg_local 0).
                // Layout [angular(3); linear(3)].
                MujocoLib.mj_objectVelocity(model, data, (int)MujocoLib.mjtObj.mjOBJ_XBODY,
                                            referenceId, velocity, 0);
                angular = new Vector3((float)velocity[0], (float)velocity[1], (float)velocity[2]);
                linear = new Vector3((float)velocity[3], (float)velocity[4], (float)velocity[5]);
            }

            for (int actionIndex = 0; actionIndex < _joints.Length; actionIndex++)
            {
                MjHingeJoint joint = _joints[actionIndex];
                _jointPositions[actionIndex] = (float)data->qpos[joint.QposAddress];
                _jointVelocities[actionIndex] = (float)data->qvel[joint.DofAddress];
            }

            var body = new CreatureBodyState(axisX, axisY, axisZ, linear, angular, _goal);
            CreatureProbe probe = ReadProbe(data, referenceId, axisZ);
            _pilot.Step(body, _jointPositions, _jointVelocities, probe, _targets);

            for (int actionIndex = 0; actionIndex < _actuators.Length; actionIndex++)
            {
                MjActuator actuator = _actuators[actionIndex];
                data->ctrl[actuator.MujocoId] = _targets[actionIndex];
                actuator.Control = _targets[actionIndex];
            }
        }

        /// <summary>
        /// Race and self-test telemetry, from mjData as well. Distances and dot products are
        /// the same in the plug-in frame and in Unity (the swap is orthogonal), so only the
        /// lead point needs converting.
        /// </summary>
        private unsafe CreatureProbe ReadProbe(MujocoLib.mjData_* data, int referenceId, Vector3 referenceUp)
        {
            int leadId = _bodies[_leadBody].MujocoId;
            Vector3 lead = PointOnBody(data, leadId, _leadOffset);
            Vector3 reference = ReadVector(data->xpos, referenceId);

            int probeA = _bodies[_pilot.ProbeBodyA].MujocoId;
            int probeB = _bodies[_pilot.ProbeBodyB].MujocoId;
            Vector3 offset = PointOnBody(data, probeB, _pilot.ProbePoint) - ReadVector(data->xpos, probeA);
            double* matrixA = data->xmat + MATRIX_SIZE * probeA;
            var probeOffset = new Vector3(
                Vector3.Dot(offset, new Vector3((float)matrixA[0], (float)matrixA[3], (float)matrixA[6])),
                Vector3.Dot(offset, new Vector3((float)matrixA[1], (float)matrixA[4], (float)matrixA[7])),
                Vector3.Dot(offset, new Vector3((float)matrixA[2], (float)matrixA[5], (float)matrixA[8])));

            return new CreatureProbe(CreatureFrames.UnityFromPlugin(lead), reference.z, referenceUp.z, probeOffset);
        }

        /// <summary>A body-frame point in the MuJoCo world: xpos + R * local.</summary>
        private static unsafe Vector3 PointOnBody(MujocoLib.mjData_* data, int bodyId, Vector3 local)
        {
            Vector3 origin = ReadVector(data->xpos, bodyId);
            double* matrix = data->xmat + MATRIX_SIZE * bodyId;
            var axisX = new Vector3((float)matrix[0], (float)matrix[3], (float)matrix[6]);
            var axisY = new Vector3((float)matrix[1], (float)matrix[4], (float)matrix[7]);
            var axisZ = new Vector3((float)matrix[2], (float)matrix[5], (float)matrix[8]);
            return origin + axisX * local.x + axisY * local.y + axisZ * local.z;
        }

        private static unsafe Vector3 ReadVector(double* array, int entry)
        {
            double* start = array + VECTOR_SIZE * entry;
            return new Vector3((float)start[0], (float)start[1], (float)start[2]);
        }
    }
}
