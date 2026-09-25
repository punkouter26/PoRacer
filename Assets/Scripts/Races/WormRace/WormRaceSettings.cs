using PoRacer.CreatureRace;
using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// Every tunable of the worm race in one asset (AGENTS.md section 5: physical constants
    /// are serialised, never buried in code). The body itself - masses, gains, limits,
    /// friction - is NOT here: it comes from worm_rig.json, the single source every trainer
    /// used, so the worms cannot drift apart through an Inspector edit.
    ///
    /// <see cref="Race"/> is the creature template's race configuration (racers, track,
    /// timing, observation, self-tests, labels), exactly what any other creature has; what
    /// follows it is the worm's own addition, the PhysX ArticulationBody lane and the
    /// cross-simulator stand-ins.
    ///
    /// Authored in the Inspector; SCN_WORM_RACE's LifetimeScope references it.
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
        /// Unity's ArticulationBody exposes no viscous joint friction. JointForceOutsideLimit
        /// applies -c*qdot through ArticulationBody.jointForce every physics step, outside the
        /// drive and its limit - the trained model, but explicit, and on these light links it
        /// diverged in every race (training/worm/RESULTS.md, problem 3).
        ///
        /// The two drive modes put the damping INSIDE the drive instead (implicit, but capped
        /// with the servo by forceLimit); DriveDampingPerRadian is the default because it is
        /// the one that races stably. Unity documents drive damping as N*m*s/rad;
        /// MujocoBiped's rung D measured it behaving per DEGREE on this Unity version
        /// (Assets/unity_export/MujocoBiped/CONTRACT.md), hence both readings.
        /// </summary>
        internal enum PassiveDamping
        {
            JointForceOutsideLimit = 0,
            DriveDampingPerRadian = 1,
            DriveDampingPerDegree = 2,
        }

        [Tooltip("The creature template's race configuration: rig, racers, track, timing, observation, self-tests.")]
        [SerializeField] private CreatureRaceConfig _race = new();

        [Header("PhysX worms")]
        [Tooltip("Friction 0.9 static and dynamic, no bounce: the worm's own and the track's.")]
        [SerializeField] private PhysicsMaterial _wormPhysicsMaterial;
        [SerializeField] private PassiveDamping _passiveDamping = PassiveDamping.DriveDampingPerRadian;
        [SerializeField] private int _solverIterations = 16;
        [SerializeField] private int _solverVelocityIterations = 4;
        [Tooltip("Fold MuJoCo's 0.01 armature into each hinge's child inertia about its axis.")]
        [SerializeField] private bool _foldArmatureIntoInertia = true;

        [Header("Both simulators")]
        [Tooltip("AGENTS rule M across simulators: every worm gets a one-way collision stand-in "
               + "in the other simulator, so a MuJoCo worm and a PhysX worm cannot pass through "
               + "each other. Worms in the same simulator collide natively.")]
        [SerializeField] private bool _crossSimulatorProxies = true;

        /// <summary>Public because WormRaceLifetimeScope registers it and the scene builder paints lanes from it.</summary>
        public CreatureRaceConfig Race => _race ??= new CreatureRaceConfig();
        internal PhysicsMaterial WormPhysicsMaterial => _wormPhysicsMaterial;
        internal PassiveDamping JointDampingMode => _passiveDamping;
        internal int SolverIterations => Mathf.Max(1, _solverIterations);
        internal int SolverVelocityIterations => Mathf.Max(1, _solverVelocityIterations);
        internal bool FoldArmatureIntoInertia => _foldArmatureIntoInertia;
        internal bool CrossSimulatorProxies => _crossSimulatorProxies;
    }
}
