using Mujoco;
using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// Physics adapter for the MuJoCo worm: reads MuJoCo's own state inside the plug-in's
    /// control callback, hands it to the worm's <see cref="WormPilot"/>, and writes the
    /// returned targets into the position servos. No decisions are made here.
    ///
    /// Everything is read straight out of mjData, in MuJoCo's frame, not off the Unity
    /// transforms the plug-in mirrors - the same choice MojucuBoyObservation made, so no
    /// frame conversion can drift between what MuJoCo simulates and what the policy sees.
    ///
    /// TIMING. MjScene.StepScene runs mj_step1, then this callback, then mj_step2. So the
    /// state read here is the state the coming step integrates from, and ctrl written here
    /// takes effect in THIS step - unlike MjActuator.Control, which the plug-in copies into
    /// ctrl only after the step (a one-step lag). Both are written: ctrl directly for
    /// zero lag, Control so the plug-in's post-step copy writes the same value back.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class MujocoWormView : MonoBehaviour
    {
        private const int VELOCITY_SIZE = 6;
        private const int MATRIX_SIZE = 9;
        private const int VECTOR_SIZE = 3;

        private readonly float[] _jointPositions = new float[WormContract.ACTION_SIZE];
        private readonly float[] _jointVelocities = new float[WormContract.ACTION_SIZE];
        private readonly float[] _targets = new float[WormContract.ACTION_SIZE];
        private readonly double[] _velocity = new double[VELOCITY_SIZE];

        private MjBody[] _segments;
        private MjHingeJoint[] _joints;
        private MjActuator[] _actuators;
        private WormPilot _pilot;
        private Vector3 _goal;
        private float _noseOffset;
        private bool _subscribed;

        internal void Bind(MjBody[] segments, MjHingeJoint[] joints, MjActuator[] actuators,
                           WormPilot pilot, Vector3 laneForwardUnity, float noseOffset)
        {
            _segments = segments;
            _joints = joints;
            _actuators = actuators;
            _pilot = pilot;
            _noseOffset = noseOffset;
            // The race direction in the plug-in's MuJoCo world frame.
            _goal = WormFrames.PluginFromUnity(laneForwardUnity);
            Subscribe();
        }

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

            int referenceId = _segments[WormContract.REFERENCE_SEGMENT].MujocoId;
            double* referenceMatrix = data->xmat + MATRIX_SIZE * referenceId;
            // xmat is row-major body-to-world: B's axis c is column c.
            var axisX = new Vector3((float)referenceMatrix[0], (float)referenceMatrix[3], (float)referenceMatrix[6]);
            var axisY = new Vector3((float)referenceMatrix[1], (float)referenceMatrix[4], (float)referenceMatrix[7]);
            var axisZ = new Vector3((float)referenceMatrix[2], (float)referenceMatrix[5], (float)referenceMatrix[8]);

            Vector3 angular;
            Vector3 linear;
            fixed (double* velocity = _velocity)
            {
                // XBODY: at the body frame origin (segment 2's centre), world orientation
                // (flg_local 0). Layout [angular(3); linear(3)].
                MujocoLib.mj_objectVelocity(model, data, (int)MujocoLib.mjtObj.mjOBJ_XBODY,
                                            referenceId, velocity, 0);
                angular = new Vector3((float)velocity[0], (float)velocity[1], (float)velocity[2]);
                linear = new Vector3((float)velocity[3], (float)velocity[4], (float)velocity[5]);
            }

            for (int actionIndex = 0; actionIndex < WormContract.ACTION_SIZE; actionIndex++)
            {
                MjHingeJoint joint = _joints[actionIndex];
                _jointPositions[actionIndex] = (float)data->qpos[joint.QposAddress];
                _jointVelocities[actionIndex] = (float)data->qvel[joint.DofAddress];
            }

            var body = new WormBodyState(axisX, axisY, axisZ, linear, angular, _goal);
            WormProbe probe = ReadProbe(data);
            _pilot.Step(body, _jointPositions, _jointVelocities, probe, _targets);

            for (int actionIndex = 0; actionIndex < WormContract.ACTION_SIZE; actionIndex++)
            {
                MjActuator actuator = _actuators[actionIndex];
                data->ctrl[actuator.MujocoId] = _targets[actionIndex];
                actuator.Control = _targets[actionIndex];
            }
        }

        /// <summary>
        /// Race and self-test telemetry, from mjData as well. Distances and dot products
        /// are the same in the plug-in frame and in Unity (the swap is orthogonal), so only
        /// the nose position needs converting.
        /// </summary>
        private unsafe WormProbe ReadProbe(MujocoLib.mjData_* data)
        {
            int headId = _segments[WormContract.HEAD_SEGMENT].MujocoId;
            int secondId = _segments[WormContract.SECOND_SEGMENT].MujocoId;
            int referenceId = _segments[WormContract.REFERENCE_SEGMENT].MujocoId;

            Vector3 head = ReadVector(data->xpos, headId);
            Vector3 second = ReadVector(data->xpos, secondId);
            Vector3 reference = ReadVector(data->xpos, referenceId);
            double* headMatrix = data->xmat + MATRIX_SIZE * headId;
            var headForward = new Vector3((float)headMatrix[0], (float)headMatrix[3], (float)headMatrix[6]);
            var headLeft = new Vector3((float)headMatrix[1], (float)headMatrix[4], (float)headMatrix[7]);
            var headUp = new Vector3((float)headMatrix[2], (float)headMatrix[5], (float)headMatrix[8]);

            Vector3 offset = second - head;
            Vector3 nose = WormFrames.UnityFromPlugin(head + headForward * _noseOffset);
            return new WormProbe(nose, reference.z, Vector3.Dot(offset, -headLeft), Vector3.Dot(offset, headUp));
        }

        private static unsafe Vector3 ReadVector(double* array, int entry)
        {
            double* start = array + VECTOR_SIZE * entry;
            return new Vector3((float)start[0], (float)start[1], (float)start[2]);
        }
    }
}
