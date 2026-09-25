using Mujoco;
using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// Stands up the MuJoCo world for one race: the MjScene singleton, the solver options
    /// the creatures trained with, and a ground plane at y = 0 (where the Unity track's
    /// collider top is, so every racer stands on the same surface without either engine
    /// knowing the other).
    ///
    /// Same shape and ordering rules as the race scene's Systems_MujocoWorld:
    ///   * MjScene is added FIRST. Every MjComponent.OnEnable reads MjScene.Instance, and
    ///     that getter creates one when none exists; adding ours second would throw
    ///     "singleton, yet multiple instances found".
    ///   * Everything MuJoCo must see is created in the same frame, before MjScene.Start
    ///     compiles the model at the start of the next frame.
    ///   * Teardown disables MjScene before anything is destroyed (see <see cref="Suspend"/>).
    ///
    /// Created in code per race rather than authored in the scene (AGENTS rule G) because
    /// MjScene compiles exactly once, from whatever Mj components exist at that moment, so it
    /// has to be born with the racers it simulates. It has no visible part; the ground you
    /// see is the authored Unity track.
    /// </summary>
    public static class MujocoCreatureWorld
    {
        /// <summary>The plane is infinite in MuJoCo; this only sizes its gizmo.</summary>
        private static readonly Vector2 GroundExtents = new(40f, 40f);

        /// <summary>org.mujoco ships mujoco.dll only (AGENTS rule F).</summary>
        public static bool IsSupported =>
            Application.platform == RuntimePlatform.WindowsEditor
            || Application.platform == RuntimePlatform.WindowsPlayer;

        /// <param name="debugFileName">File name under temporaryCachePath for the generated MJCF; empty = none.</param>
        public static GameObject Build(int solverIterations, string debugFileName, CreatureContact floor)
        {
            var world = new GameObject("MuJoCoWorld");
            world.AddComponent<MjScene>();

            MjGlobalSettings globals = world.AddComponent<MjGlobalSettings>();
            // The trainers' <option>. Timestep and gravity are not set here on purpose: the
            // plug-in always takes them from Time.fixedDeltaTime and Physics.gravity.
            MjOptionStruct options = globals.GlobalOptions;
            options.Integrator = IntegratorType.implicitfast;
            options.Solver = ConstraintSolverType.Newton;
            options.Iterations = solverIterations;
            options.Cone = FrictionConeType.pyramidal;
            globals.GlobalOptions = options;
            if (!string.IsNullOrEmpty(debugFileName))
            {
                globals.DebugFileName = debugFileName;
            }

            var ground = new GameObject("MuJoCoGround");
            ground.transform.SetParent(world.transform, false);
            ground.transform.localPosition = Vector3.zero;
            ground.transform.localRotation = Quaternion.identity;
            MjGeom plane = ground.AddComponent<MjGeom>();
            plane.ShapeType = MjShapeComponent.ShapeTypes.Plane;
            plane.Plane.Extents = GroundExtents;
            ApplyContact(plane, floor);
            return world;
        }

        /// <summary>A rig's contact parameters onto one plug-in geom.</summary>
        public static void ApplyContact(MjGeom geom, CreatureContact contact)
        {
            MjGeomSettings settings = geom.Settings;
            settings.Friction.Sliding = contact.Friction.x;
            settings.Friction.Torsional = contact.Friction.y;
            settings.Friction.Rolling = contact.Friction.z;
            settings.Solver.ConDim = contact.ConDim;
            settings.Solver.SolRef.TimeConst = contact.SolRef.x;
            settings.Solver.SolRef.DampRatio = contact.SolRef.y;
            settings.Solver.SolImp.DMin = contact.SolImp.x;
            settings.Solver.SolImp.DMax = contact.SolImp.y;
            settings.Solver.SolImp.Width = contact.SolImp.z;
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
