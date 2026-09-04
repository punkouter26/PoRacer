using Mujoco;
using UnityEngine;

namespace PoRacer.Systems
{
    /// <summary>
    /// Owns the single MuJoCo world a race needs when Fido is on the grid.
    ///
    /// MuJoCo is not a per-creature simulation: <see cref="MjScene"/> is a singleton that
    /// compiles one model out of every Mj* component in the scene and steps it once per
    /// FixedUpdate. So the world is built once, after the spawner has placed every racer,
    /// and torn down with them. Two consequences worth knowing:
    ///
    ///   * Every Fido on the grid shares one MuJoCo model, so Fidos collide with each
    ///     other and with the ground the way MuJoCo intends. They do not collide with the
    ///     PhysX racers, which live in a different solver entirely.
    ///   * <see cref="MjScene.CreateScene"/> reads Unity transforms to author the model,
    ///     so racers must be positioned before <see cref="Build"/> runs. That is why this
    ///     is called at the end of the spawn loop rather than alongside each Instantiate.
    ///
    /// The ground plane and the solver options live here rather than in the Fido prefab:
    /// they are properties of the world, and eight prefabs each carrying a floor would
    /// stack eight coincident planes.
    /// </summary>
    internal static class Systems_MujocoWorld
    {
        /// <summary>
        /// MuJoCo's plane is infinite regardless of this, which only sizes the gizmo and
        /// the generated preview mesh. Kept near the track's own footprint.
        /// </summary>
        private static readonly Vector2 GroundExtents = new(120f, 120f);

        // Straight from creature.xml's <geom> default, which is what Fido trained against.
        private const float GROUND_FRICTION_SLIDING = 0.8f;
        private const float GROUND_FRICTION_TORSIONAL = 0.02f;
        private const float GROUND_FRICTION_ROLLING = 0.01f;
        private const int GROUND_CONDIM = 3;

        private static GameObject _world;

        /// <summary>True while a MuJoCo world is standing.</summary>
        internal static bool Exists => _world != null;

        /// <summary>
        /// The plug-in ships mujoco.dll and nothing else, so MuJoCo creatures can only
        /// step on Windows; anywhere else every call is a DllNotFoundException.
        /// </summary>
        internal static bool IsSupported =>
            Application.platform == RuntimePlatform.WindowsEditor
            || Application.platform == RuntimePlatform.WindowsPlayer;

        /// <summary>The roster gate: a brain to race with, and a simulator that runs here.</summary>
        internal static bool CanRace(PoRacer.Models.CreatureCatalog.CreatureEntry entry)
        {
            if (entry.prefab == null || !entry.HasBrain)
            {
                return false;
            }
            return IsSupported || entry.prefab.GetComponentInChildren<PoRacer.Agents.IMujocoCreature>(true) == null;
        }

        /// <summary>
        /// Stands up the MuJoCo world: solver options matching training, a ground plane at
        /// y = 0, and the MjScene that compiles them together with every Fido already in
        /// the scene. Safe to call when no Fido is racing — the caller decides that; this
        /// only guards against building twice.
        /// </summary>
        internal static void Build()
        {
            if (_world != null || !IsSupported)
            {
                return;
            }

            // Deliberately unparented. The obvious home would be the track root, but that
            // is rebuilt for every race, and taking the MjScene down mid-race would free
            // the native model out from under the creatures stepping it. Despawn owns the
            // teardown instead.
            _world = new GameObject("MuJoCoWorld");

            // MjScene first, and this order is not cosmetic. Every MjComponent's OnEnable
            // reads MjScene.Instance, and that getter *creates* an MjScene when none
            // exists. Add the ground geom first and it conjures its own "MjScene" object;
            // ours is then the second and MjScene.Awake throws "singleton, yet multiple
            // instances found", leaving two half-built worlds. Claiming the singleton here
            // means everything added below simply finds it.
            _world.AddComponent<MjScene>();

            ConfigureOptions(_world.AddComponent<MjGlobalSettings>());
            BuildGround(_world.transform);

            // MjScene.Start compiles the model at the end of this frame, by which time the
            // options, the ground and the racers are all in place. Racers added later still
            // arrive safely: MjComponent.OnEnable flags SceneRecreationAtLateUpdateRequested
            // whenever a model already exists, so the plug-in rebuilds around them.
        }

        /// <summary>
        /// Destroys the world. MjScene.OnDestroy frees the native model and data, and
        /// although the plug-in never nulls its own static instance, Unity's overloaded
        /// == reports the destroyed component as null, so the next race's MjScene claims
        /// the singleton cleanly.
        /// </summary>
        internal static void Teardown()
        {
            if (_world == null)
            {
                return;
            }
            Object.Destroy(_world);
            _world = null;
        }

        /// <summary>
        /// Mirrors creature.xml's &lt;option&gt;, which the importer does not carry onto a
        /// prefab. Timestep is deliberately absent: the plug-in always takes it from
        /// Time.fixedDeltaTime, and Fido's decimation is set against the project's rate
        /// instead. ls_iterations is absent too — MjOptionStruct has no field for it, so
        /// MuJoCo uses its default 50 where training used 8, i.e. it solves more
        /// accurately than training did.
        /// </summary>
        private static void ConfigureOptions(MjGlobalSettings settings)
        {
            MjOptionStruct options = settings.GlobalOptions;
            options.Integrator = IntegratorType.implicitfast;
            options.Solver = ConstraintSolverType.Newton;
            options.Iterations = 4;
            options.Cone = FrictionConeType.pyramidal;
            settings.GlobalOptions = options;
        }

        /// <summary>
        /// The MuJoCo ground, at y = 0 — exactly where Systems_TrackBuilder puts the top
        /// of the flat track's collider slab, so Fido and the PhysX racers stand on the
        /// same surface without either engine knowing about the other.
        /// </summary>
        private static void BuildGround(Transform parent)
        {
            var ground = new GameObject("MuJoCoGround");
            ground.transform.SetParent(parent, false);
            ground.transform.localPosition = Vector3.zero;
            ground.transform.localRotation = Quaternion.identity;

            var geom = ground.AddComponent<MjGeom>();
            geom.ShapeType = MjShapeComponent.ShapeTypes.Plane;
            geom.Plane.Extents = GroundExtents;

            MjGeomSettings settings = geom.Settings;
            settings.Friction.Sliding = GROUND_FRICTION_SLIDING;
            settings.Friction.Torsional = GROUND_FRICTION_TORSIONAL;
            settings.Friction.Rolling = GROUND_FRICTION_ROLLING;
            settings.Solver.ConDim = GROUND_CONDIM;
            geom.Settings = settings;

            // The track already draws a ground; this one only needs to exist for MuJoCo.
            // MjGeom's own mesh preview is added by the importer, not by AddComponent, so
            // there is nothing to hide here beyond leaving the renderer off.
        }
    }
}
