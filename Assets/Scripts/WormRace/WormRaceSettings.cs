using Unity.InferenceEngine;
using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// Every tunable of the worm race in one asset (AGENTS.md section 5: physical constants
    /// are serialised, never buried in code). The body itself - masses, gains, limits,
    /// friction - is NOT here: it comes from worm_rig.json, the single source both trainers
    /// used, so the two worms cannot drift apart through an Inspector edit.
    ///
    /// Built and wired by Editor_BuildWormRaceScene; the defaults below are the race spec.
    /// </summary>
    [CreateAssetMenu(menuName = "PoRacer/Worm Race Settings", fileName = "WormRaceSettings")]
    public sealed class WormRaceSettings : ScriptableObject
    {
        /// <summary>
        /// Where the PhysX worm's joint damping (worm_rig.json jointDamping) lives.
        ///
        /// WORM_SPEC "Details resolved" 16: in both trainers the damping is a PASSIVE
        /// -c*qdot that the servo's force limit does not cap (MuJoCo joint damping; PhysX
        /// joint viscous friction in Isaac Lab), and the servo itself is kp with no damping.
        /// Unity's ArticulationBody exposes no viscous joint friction, so the default applies
        /// -c*qdot through ArticulationBody.jointForce every physics step, outside the drive
        /// and its limit - the trained model. That is explicit; at c = 2 and dt = 5 ms it is
        /// well inside the stable range for these link inertias (see INSTALL.md).
        ///
        /// The two drive modes put the damping INSIDE the drive instead (implicit, but capped
        /// with the servo by forceLimit). Unity documents drive damping as N*m*s/rad;
        /// MujocoBiped's rung D measured it behaving per DEGREE on this Unity version
        /// (Assets/unity_export/MujocoBiped/CONTRACT.md), hence both readings.
        /// </summary>
        internal enum PassiveDamping
        {
            JointForceOutsideLimit = 0,
            DriveDampingPerRadian = 1,
            DriveDampingPerDegree = 2,
        }

        [Header("Body and brains")]
        [Tooltip("worm_rig.json copied from training/worm/. Both worms are built from it.")]
        [SerializeField] private TextAsset _rigJson;
        [Tooltip("training/worm/export/worm_mujoco.onnx, imported as a ModelAsset.")]
        [SerializeField] private ModelAsset _mujocoBrain;
        [Tooltip("training/worm/export/worm_isaac.onnx, imported as a ModelAsset.")]
        [SerializeField] private ModelAsset _isaacBrain;

        [Header("Look (AGENTS rule D: red and green are reserved)")]
        [SerializeField] private Material _mujocoMaterial;
        [SerializeField] private Material _isaacMaterial;
        [SerializeField] private Color _mujocoColor = new(0.16f, 0.45f, 0.95f);
        [SerializeField] private Color _isaacColor = new(1.0f, 0.55f, 0.10f);

        [Header("Track")]
        [SerializeField] private float _trackLength = 20f;
        [SerializeField] private float _laneSpacing = 2f;
        [SerializeField] private float _startLineZ;
        [Tooltip("The nose starts this far behind the start line.")]
        [SerializeField] private float _startGap = 0.01f;

        [Header("Race")]
        [SerializeField] private int _countdownSeconds = 3;
        [SerializeField] private float _timeLimitSeconds = 60f;
        [SerializeField] private float _resultsHoldSeconds = 4f;
        [SerializeField] private int _seriesLength = 5;
        [Tooltip("Start a series on play. The editor harness overrides count and mode.")]
        [SerializeField] private bool _autoStartSeries = true;
        [SerializeField] private float _selfTestSettleSeconds = 1f;

        [Header("PhysX (Isaac) worm")]
        [Tooltip("Friction 0.9 static and dynamic, no bounce: the worm's own and the track's.")]
        [SerializeField] private PhysicsMaterial _wormPhysicsMaterial;
        [SerializeField] private PassiveDamping _passiveDamping = PassiveDamping.JointForceOutsideLimit;
        [SerializeField] private int _solverIterations = 16;
        [SerializeField] private int _solverVelocityIterations = 4;
        [Tooltip("Fold MuJoCo's 0.01 armature into each hinge's child inertia about its axis.")]
        [SerializeField] private bool _foldArmatureIntoInertia = true;

        [Header("MuJoCo worm")]
        [Tooltip("worm.xml uses 10. ls_iterations (8) has no plug-in field; MuJoCo uses 50.")]
        [SerializeField] private int _mujocoSolverIterations = 10;
        [Tooltip("Write the MJCF the plug-in generates to Application.temporaryCachePath, "
               + "so it can be diffed against training/worm/worm.xml.")]
        [SerializeField] private bool _dumpMujocoMjcf = true;

        [Header("Both")]
        [Tooltip("AGENTS rule M across simulators: each worm gets a one-way collision stand-in "
               + "in the other's physics, so they cannot pass through each other.")]
        [SerializeField] private bool _crossSimulatorProxies = true;
        [Tooltip("Feed back the clipped action as 'previous action' (WORM_SPEC: actions are "
               + "clipped to [-1, 1]). Untick if a trainer recorded the raw network output.")]
        [SerializeField] private bool _previousActionClipped = true;

        internal TextAsset RigJson => _rigJson;
        internal ModelAsset MujocoBrain => _mujocoBrain;
        internal ModelAsset IsaacBrain => _isaacBrain;
        internal Material MujocoMaterial => _mujocoMaterial;
        internal Material IsaacMaterial => _isaacMaterial;
        internal Color MujocoColor => _mujocoColor;
        internal Color IsaacColor => _isaacColor;
        public float TrackLength => Mathf.Max(1f, _trackLength);
        public float LaneSpacing => Mathf.Max(1f, _laneSpacing);
        public float StartLineZ => _startLineZ;
        internal float StartGap => Mathf.Max(0f, _startGap);
        internal int CountdownSeconds => Mathf.Max(0, _countdownSeconds);
        internal float TimeLimitSeconds => Mathf.Max(1f, _timeLimitSeconds);
        internal float ResultsHoldSeconds => Mathf.Max(0f, _resultsHoldSeconds);
        internal int SeriesLength => Mathf.Max(1, _seriesLength);
        internal bool AutoStartSeries => _autoStartSeries;
        internal float SelfTestSettleSeconds => Mathf.Max(0f, _selfTestSettleSeconds);
        internal PhysicsMaterial WormPhysicsMaterial => _wormPhysicsMaterial;
        internal PassiveDamping JointDampingMode => _passiveDamping;
        internal int SolverIterations => Mathf.Max(1, _solverIterations);
        internal int SolverVelocityIterations => Mathf.Max(1, _solverVelocityIterations);
        internal bool FoldArmatureIntoInertia => _foldArmatureIntoInertia;
        internal int MujocoSolverIterations => Mathf.Max(1, _mujocoSolverIterations);
        internal bool DumpMujocoMjcf => _dumpMujocoMjcf;
        internal bool CrossSimulatorProxies => _crossSimulatorProxies;
        internal bool PreviousActionClipped => _previousActionClipped;

        /// <summary>
        /// Lane centre, x offset from the track centre. Public because
        /// Editor_BuildWormRaceScene lays the painted lines out from these same numbers.
        /// </summary>
        public float LaneX(int lane)
        {
            return (lane - 0.5f) * LaneSpacing;
        }
    }
}
