using System;
using System.Collections.Generic;
using Mujoco;
using UnityEngine;
#if CREATURE_HAS_INFERENCE
using Unity.InferenceEngine;
#endif

namespace Creature.MojucuBoy
{
    /// <summary>
    /// Runtime controller for MojucuBoy, the 21-DOF MuJoCo humanoid racer.
    ///
    /// Drives the policy exported by training/mojucuboy/export_onnx.py. Observations are
    /// built by <see cref="MojucuBoyObservation"/> from MuJoCo's own mjData -- the same
    /// code path Gate 3 proved equal to the Python trainer to 0.000e+00 -- and the
    /// action is written straight back into the MjActuator controls.
    ///
    /// Timing mirrors training exactly: MuJoCo steps at Time.fixedDeltaTime
    /// (0.005 s) and the policy runs every <see cref="DECIMATION"/> = 4 steps, so
    /// 0.02 s, the rate the policy was trained at. If the project's fixed timestep
    /// ever changes, this is one of the places that has to change with it.
    ///
    /// The observation normaliser lives INSIDE the ONNX graph, so there are no
    /// normalisation statistics on this side to get wrong: raw observations in,
    /// actions out.
    /// </summary>
    [DefaultExecutionOrder(10)]   // after MjScene.Awake (order 0) claims the singleton
    [DisallowMultipleComponent]
    public sealed class MojucuBoyController : MonoBehaviour
    {
        public const int DECIMATION = 4;

        /// <summary>Fraction of each joint's half-range a saturated action spans,
        /// measured from the standing stance. Must match ACTION_SCALE in
        /// training/mojucuboy/mojucuboy_env.py -- it is part of the trained contract.</summary>
        public const float ACTION_SCALE = 0.6f;

#if CREATURE_HAS_INFERENCE
        [Tooltip("MojucuBoy_v01.onnx, exported by training/mojucuboy/export_onnx.py.")]
        [SerializeField] private ModelAsset _modelAsset;
#endif

        [Tooltip("mojucuboy_rig.json: actuator order, joint ranges and the standing stance.")]
        [SerializeField] private TextAsset _rigJson;

        [Tooltip("Hold the standing stance instead of running the policy. For A/B tests.")]
        [SerializeField] private bool _passive;

        private readonly List<MjActuator> _actuators = new();
        private readonly List<MjHingeJoint> _joints = new();
        private int[] _qposAddr;
        private int[] _dofAddr;
        private float[] _stance;
        private float[] _rangeLo;
        private float[] _rangeHi;
        private float[] _obs;
        private float[] _action;
        private int _rootBodyId = -1;
        private int _stepCounter;
        private bool _bound;
        private bool _failed;

        private float _commandHeading;
        private float _commandSpeed = 1.5f;

#if CREATURE_HAS_INFERENCE
        private Worker _worker;
        private Tensor<float> _input;
#endif

        /// <summary>Steer the racer toward a world-space point. Unlike Fido, whose
        /// 33 observations describe only his own body, Boy carries a heading command
        /// in his observation and can actually be steered to a finish line.</summary>
        public void SetGoal(Vector3 worldTarget)
        {
            // Unity +Z is MuJoCo +Y and Unity +X is MuJoCo +X, so a planar heading
            // maps across as atan2(unityZ, unityX). See MjEngineTool.UnityVector3.
            Vector3 here = transform.position;
            _commandHeading = Mathf.Atan2(worldTarget.z - here.z, worldTarget.x - here.x);
        }

        private void OnEnable()
        {
            if (!MjScene.InstanceExists)
            {
                Debug.LogError($"[{name}] no MjScene in the scene; MojucuBoyController needs one.", this);
                _failed = true;
                enabled = false;
                return;
            }
            MjScene.Instance.ctrlCallback += OnControl;
        }

        private void OnDisable()
        {
            if (MjScene.InstanceExists)
            {
                MjScene.Instance.ctrlCallback -= OnControl;
            }
        }

        private void OnDestroy() => ReleaseWorker();

        private void ReleaseWorker()
        {
#if CREATURE_HAS_INFERENCE
            _worker?.Dispose();
            _worker = null;
            _input?.Dispose();
            _input = null;
#endif
        }

        private unsafe void OnControl(object sender, MjStepArgs e)
        {
            if (_failed)
            {
                return;
            }
            if (!_bound)
            {
                if (!TryBind(e.model))
                {
                    _failed = true;
                    return;
                }
                _bound = true;
                CheckTiming();
            }

            if (_passive || !HasBrain)
            {
                WriteStance();
                return;
            }

            if (_stepCounter % DECIMATION == 0)
            {
                MojucuBoyObservation.Build(e.data, _rootBodyId, _qposAddr, _dofAddr,
                                     _commandHeading, _commandSpeed, _action, _obs);
                Evaluate();
            }
            _stepCounter++;
            WriteAction();
        }

        /// <summary>
        /// Points the heading command along the racer's current facing, so "no goal"
        /// means "carry straight on" rather than "turn toward world +X".
        /// </summary>
        private void AlignCommandToFacing()
        {
            Transform body = _joints.Count > 0 ? transform : transform;
            Vector3 forward = body.forward;
            // Unity +Z is MuJoCo +Y and Unity +X is MuJoCo +X, so a planar heading maps
            // across as atan2(unityZ, unityX). Same mapping SetGoal uses.
            _commandHeading = Mathf.Atan2(forward.z, forward.x);
        }

        private bool HasBrain
        {
#if CREATURE_HAS_INFERENCE
            get => _worker != null;
#else
            get => false;
#endif
        }

        private unsafe bool TryBind(MujocoLib.mjModel_* model)
        {
            if (_rigJson == null)
            {
                Debug.LogError($"[{name}] no rig JSON assigned.", this);
                return false;
            }
            MojucuBoyRig rig = MojucuBoyRig.Parse(_rigJson.text);
            int n = rig.ActuatorOrder.Length;
            if (n != MojucuBoyObservation.JOINT_COUNT)
            {
                Debug.LogError($"[{name}] rig has {n} joints, expected "
                             + $"{MojucuBoyObservation.JOINT_COUNT}.", this);
                return false;
            }

            // Resolve ids from the components' OWN MujocoId, never mj_name2id.
            //
            // The name lookup works in a scene this creature's own setup built, where
            // MjGlobalSettings.UseRawGameObjectNames keeps the authored names. It does
            // NOT work in the race scene: Systems_MujocoWorld owns the MjGlobalSettings
            // there and leaves the default naming, so every element is exported as
            // "<name>_<n>" -- and with more than one MuJoCo racer on the grid the bare
            // names would be ambiguous anyway. Names still match components, but only
            // WITHIN this racer's own hierarchy, where the prefab keeps them unique.
            var bodies = GetComponentsInChildren<MjBody>(true);
            MjBody rootBody = Find(bodies, rig.RootBody);
            if (rootBody == null || rootBody.MujocoId < 0)
            {
                Debug.LogError($"[{name}] root body '{rig.RootBody}' missing, or not bound "
                             + "to the compiled model yet.", this);
                return false;
            }
            _rootBodyId = rootBody.MujocoId;

            var actuators = GetComponentsInChildren<MjActuator>(true);
            var joints = GetComponentsInChildren<MjHingeJoint>(true);
            _qposAddr = new int[n];
            _dofAddr = new int[n];
            _actuators.Clear();
            _joints.Clear();

            for (int i = 0; i < n; i++)
            {
                string jointName = rig.ActuatorOrder[i];
                MjHingeJoint joint = Find(joints, jointName);
                if (joint == null || joint.MujocoId < 0)
                {
                    Debug.LogError($"[{name}] joint '{jointName}' missing, or not bound.", this);
                    return false;
                }
                _qposAddr[i] = model->jnt_qposadr[joint.MujocoId];
                _dofAddr[i] = model->jnt_dofadr[joint.MujocoId];

                MjActuator actuator = Find(actuators, "act_" + jointName);
                if (actuator == null)
                {
                    Debug.LogError($"[{name}] actuator 'act_{jointName}' not found.", this);
                    return false;
                }
                _actuators.Add(actuator);
                _joints.Add(joint);
            }

            _stance = rig.Stance;
            _rangeLo = rig.RangeLo;
            _rangeHi = rig.RangeHi;
            _obs = new float[MojucuBoyObservation.OBS_SIZE];
            _action = new float[MojucuBoyObservation.ACTION_SIZE];

            // Start the heading command pointing where he ALREADY faces.
            //
            // Left at its default of 0 the command means "walk toward world +X", and he
            // spawns facing +Z -- so his very first observation reports a 90 degree
            // heading error, nearly three times the +/-34 degree envelope he was trained
            // in. Measured in the race scene: he went down inside 5 seconds, before the
            // spawner's goal was ever applied. A racer with no goal must walk straight
            // ahead, not be told to turn ninety degrees.
            AlignCommandToFacing();

#if CREATURE_HAS_INFERENCE
            if (_modelAsset != null)
            {
                Model runtimeModel = ModelLoader.Load(_modelAsset);
                // CPU backend deliberately, matching the other racers in this project.
                _worker = new Worker(runtimeModel, BackendType.CPU);
                _input = new Tensor<float>(new TensorShape(1, MojucuBoyObservation.OBS_SIZE));
            }
            else
            {
                Debug.LogWarning($"[{name}] no ModelAsset assigned; holding the stance.", this);
            }
#else
            Debug.LogWarning($"[{name}] built without the Inference Engine define; "
                           + "holding the stance. Add com.unity.ai.inference to the "
                           + "Creature asmdef versionDefines.", this);
#endif
            return true;
        }

        private static T Find<T>(T[] items, string wanted) where T : Component
        {
            for (int i = 0; i < items.Length; i++)
            {
                if (items[i].name == wanted)
                {
                    return items[i];
                }
            }
            return null;
        }

        private void Evaluate()
        {
#if CREATURE_HAS_INFERENCE
            _input.Upload(_obs);
            _worker.Schedule(_input);
            var output = _worker.PeekOutput() as Tensor<float>;
            output.CompleteAllPendingOperations();
            for (int i = 0; i < MojucuBoyObservation.ACTION_SIZE; i++)
            {
                _action[i] = Mathf.Clamp(output[0, i], -1f, 1f);
            }
#endif
        }

        private void WriteAction()
        {
            for (int i = 0; i < _actuators.Count; i++)
            {
                float half = 0.5f * (_rangeHi[i] - _rangeLo[i]);
                float target = _stance[i] + ACTION_SCALE * half * _action[i];
                _actuators[i].Control = Mathf.Clamp(target, _rangeLo[i], _rangeHi[i]);
            }
        }

        private void WriteStance()
        {
            for (int i = 0; i < _actuators.Count; i++)
            {
                _actuators[i].Control = _stance[i];
            }
        }

        /// <summary>
        /// The MuJoCo Unity plug-in ignores the MJCF's &lt;option timestep&gt; and uses
        /// Time.fixedDeltaTime instead. If that does not match what the policy trained
        /// with, the racer is controlled at the wrong rate and moves badly for no
        /// visible reason. Catch it loudly rather than letting it degrade silently.
        /// </summary>
        private void CheckTiming()
        {
            const float TRAINED_PHYSICS_DT = 0.005f;
            float actual = Time.fixedDeltaTime;
            if (Mathf.Abs(actual - TRAINED_PHYSICS_DT) > 1e-4f)
            {
                Debug.LogError(
                    $"[{name}] Time.fixedDeltaTime is {actual:F5} s but the policy was "
                  + $"trained at {TRAINED_PHYSICS_DT:F5} s with decimation {DECIMATION} "
                  + $"({TRAINED_PHYSICS_DT * DECIMATION:F3} s policy step). Either restore "
                  + "the project timestep or retrain; the racer will not walk correctly.",
                    this);
            }
        }
    }
}
