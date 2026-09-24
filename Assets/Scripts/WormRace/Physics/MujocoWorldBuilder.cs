// SUPERSEDED by Assets/Scripts/CreatureRace/Mujoco/MujocoCreatureWorld.cs (creature template, 2026-09-24).
// Kept out of compilation, content unchanged, until someone deletes this file and its .meta.
#if PORACER_WORMRACE_LEGACY
using Mujoco;
using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// Stands up the MuJoCo world for one race: the MjScene singleton, the solver options
    /// from worm.xml, and a ground plane at y = 0 (where the Unity track's collider top is,
    /// so every worm lies on the same surface without either engine knowing the other).
    ///
    /// Same shape and same ordering rules as the race scene's Systems_MujocoWorld:
    ///   * MjScene is added FIRST. Every MjComponent.OnEnable reads MjScene.Instance, and
    ///     that getter creates one when none exists; adding ours second would throw
    ///     "singleton, yet multiple instances found".
    ///   * Everything MuJoCo must see is created in the same frame, before MjScene.Start
    ///     compiles the model at the start of the next frame.
    ///   * Teardown disables MjScene before anything is destroyed (see <see cref="Suspend"/>).
    ///
    /// Created in code per race rather than authored in the scene (AGENTS rule G) for the
    /// same reason as Systems_MujocoWorld: MjScene compiles exactly once, from whatever
    /// Mj components exist at that moment, so it has to be born with the worms it simulates
    /// (every MuJoCo racer of the race, plus the PhysX worms' mocap stand-ins).
    /// It has no visible part; the ground you see is the authored Unity track.
    /// </summary>
    internal static class MujocoWorldBuilder
    {
        /// <summary>File name under Application.temporaryCachePath for the generated MJCF.</summary>
        public const string DEBUG_MJCF_FILE = "wormrace_mujoco_scene.xml";

        // worm.xml <default><geom>: friction="0.9 0.005 0.0001" condim="3" solref="0.01 1"
        // solimp="0.9 0.95 0.001". Sliding friction comes from the rig; the rest is here.
        private const float TORSIONAL_FRICTION = 0.005f;
        private const float ROLLING_FRICTION = 0.0001f;
        private const int CONTACT_DIMENSION = 3;
        private const float CONTACT_TIME_CONSTANT = 0.01f;
        private const float CONTACT_DAMPING_RATIO = 1f;
        private const float IMPEDANCE_MIN = 0.9f;
        private const float IMPEDANCE_MAX = 0.95f;
        private const float IMPEDANCE_WIDTH = 0.001f;

        /// <summary>The plane is infinite in MuJoCo; this only sizes its gizmo.</summary>
        private static readonly Vector2 GroundExtents = new(40f, 40f);

        /// <summary>org.mujoco ships mujoco.dll only (AGENTS rule F).</summary>
        public static bool IsSupported =>
            Application.platform == RuntimePlatform.WindowsEditor
            || Application.platform == RuntimePlatform.WindowsPlayer;

        public static GameObject Build(int solverIterations, bool dumpMjcf, float slidingFriction)
        {
            var world = new GameObject("MuJoCoWorld");
            world.AddComponent<MjScene>();

            MjGlobalSettings globals = world.AddComponent<MjGlobalSettings>();
            // worm.xml <option>. Timestep and gravity are not set here on purpose: the
            // plug-in always takes them from Time.fixedDeltaTime and Physics.gravity.
            MjOptionStruct options = globals.GlobalOptions;
            options.Integrator = IntegratorType.implicitfast;
            options.Solver = ConstraintSolverType.Newton;
            options.Iterations = solverIterations;
            options.Cone = FrictionConeType.pyramidal;
            globals.GlobalOptions = options;
            if (dumpMjcf)
            {
                globals.DebugFileName = DEBUG_MJCF_FILE;
            }

            var ground = new GameObject("MuJoCoGround");
            ground.transform.SetParent(world.transform, false);
            ground.transform.localPosition = Vector3.zero;
            ground.transform.localRotation = Quaternion.identity;
            MjGeom plane = ground.AddComponent<MjGeom>();
            plane.ShapeType = MjShapeComponent.ShapeTypes.Plane;
            plane.Plane.Extents = GroundExtents;
            ApplyContact(plane, slidingFriction);
            return world;
        }

        /// <summary>worm.xml's geom contact parameters, for the floor, the worm and the proxies.</summary>
        public static void ApplyContact(MjGeom geom, float slidingFriction)
        {
            MjGeomSettings settings = geom.Settings;
            settings.Friction.Sliding = slidingFriction;
            settings.Friction.Torsional = TORSIONAL_FRICTION;
            settings.Friction.Rolling = ROLLING_FRICTION;
            settings.Solver.ConDim = CONTACT_DIMENSION;
            settings.Solver.SolRef.TimeConst = CONTACT_TIME_CONSTANT;
            settings.Solver.SolRef.DampRatio = CONTACT_DAMPING_RATIO;
            settings.Solver.SolImp.DMin = IMPEDANCE_MIN;
            settings.Solver.SolImp.DMax = IMPEDANCE_MAX;
            settings.Solver.SolImp.Width = IMPEDANCE_WIDTH;
            geom.Settings = settings;
        }

        /// <summary>
        /// Stops stepping immediately. Call BEFORE destroying anything MuJoCo simulates:
        /// Destroy is deferred to the end of the frame, every dying MjComponent requests a
        /// scene rebuild from OnDisable, and a rebuild racing the teardown is the native
        /// crash Systems_MujocoWorld.Suspend documents. A disabled MjScene runs neither
        /// FixedUpdate nor LateUpdate; its OnDestroy still frees the native model.
        /// </summary>
        public static void Suspend(GameObject world)
        {
            if (world == null)
            {
                return;
            }
            MjScene scene = world.GetComponent<MjScene>();
            if (scene != null)
            {
                scene.enabled = false;
            }
        }
    }
}
#endif
