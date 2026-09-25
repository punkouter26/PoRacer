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
        private static bool _suspended;
        private static bool _startHeld;
        private static Views.RaceCourseView _pendingCourse;

        /// <summary>
        /// How thick each road slab is. Only the top face is stood on; the depth exists so
        /// a foot landing slightly inside a slab is pushed out rather than passing through.
        /// </summary>
        private const float ROAD_SLAB_THICKNESS = 0.6f;

        /// <summary>
        /// Each slab is stretched this much past its centreline segment.
        ///
        /// Butted end to end, slabs leave a wedge of gap on the OUTSIDE of every bend -
        /// at Acrobat's 26 deg per segment and a 3 m half-width that is about
        /// 3 * tan(13 deg) = 0.7 m of hole, which is plenty for a foot to drop through.
        /// Overlapping costs nothing: MuJoCo static geoms do not collide with each other.
        /// </summary>
        private const float ROAD_SLAB_OVERLAP = 1.25f;

        /// <summary>
        /// The catch-plane under a course. Far below the road, NOT at y = 0: at y = 0 a
        /// racer that fell off would land on an invisible floor and keep running there,
        /// which is exactly the bug that made MuJoCo racers unraceable on courses in the
        /// first place. Down here a fall reads as a fall.
        /// </summary>
        private const float COURSE_CATCH_PLANE_Y = -50f;

        /// <summary>True while a MuJoCo world is standing and still stepping.</summary>
        internal static bool Exists => _world != null && !_suspended;

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
        /// <summary>
        /// Stands up the world with an authored course mirrored into it, so a MuJoCo racer
        /// can run a course at all. Pass null for a builder map and it behaves exactly as
        /// the parameterless <see cref="Build()"/> always did.
        /// </summary>
        internal static void Build(Views.RaceCourseView course)
        {
            _pendingCourse = course;
            Build();
            _pendingCourse = null;
        }

        /// <summary>
        /// As <see cref="Build(Views.RaceCourseView)"/>, plus every solid box and sphere
        /// collider under <paramref name="scenery"/> mirrored into MuJoCo, so a MuJoCo
        /// racer is stopped by the same barriers, stands and props as the PhysX racers
        /// (AGENTS rule M) instead of walking through them on its own infinite plane.
        /// </summary>
        internal static void Build(Views.RaceCourseView course, Transform scenery)
        {
            bool fresh = _world == null;
            Build(course);
            if (fresh && _world != null && scenery != null)
            {
                MirrorSceneryColliders(_world.transform, scenery);
            }
        }

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
            _suspended = false;
            _startHeld = false;

            // MjScene first, and this order is not cosmetic. Every MjComponent's OnEnable
            // reads MjScene.Instance, and that getter *creates* an MjScene when none
            // exists. Add the ground geom first and it conjures its own "MjScene" object;
            // ours is then the second and MjScene.Awake throws "singleton, yet multiple
            // instances found", leaving two half-built worlds. Claiming the singleton here
            // means everything added below simply finds it.
            _world.AddComponent<MjScene>();

            ConfigureOptions(_world.AddComponent<MjGlobalSettings>());
            if (_pendingCourse != null)
            {
                // A course gets a road built out of box geoms plus a catch-plane far
                // below, instead of a plane at y = 0 he would simply stand on.
                BuildCourseRoad(_world.transform, _pendingCourse);
                BuildGround(_world.transform, COURSE_CATCH_PLANE_Y);
            }
            else
            {
                BuildGround(_world.transform, 0f);
            }

            // MjScene.Start compiles the model at the end of this frame, by which time the
            // options, the ground and the racers are all in place. Racers added later still
            // arrive safely: MjComponent.OnEnable flags SceneRecreationAtLateUpdateRequested
            // whenever a model already exists, so the plug-in rebuilds around them.
        }

        /// <summary>
        /// Freezes or resumes the MuJoCo step for the pre-race countdown.
        ///
        /// This is the start gate for every MuJoCo racer, because nothing else can hold
        /// one: they have no ArticulationBody to pin and MuJoCo, not Unity, writes their
        /// transforms. Not stepping leaves them exactly as the spawner placed them —
        /// the trained stance, frozen — which is a better hold than pinning gives the
        /// PhysX racers.
        ///
        /// Call it only after the model has compiled. MjScene compiles in Start, and a
        /// disabled component's Start never runs, so holding in the same frame as
        /// <see cref="Build"/> would defer compilation to the release instead of
        /// freezing anything. The spawner holds at the top of the countdown, several
        /// frames later, which is safely past that.
        ///
        /// <see cref="Suspend"/> outranks this: once the world is being torn down it
        /// must stay stopped, so a release cannot restart it.
        /// </summary>
        internal static void HoldStepping(bool held)
        {
            if (_world == null || _suspended || _startHeld == held)
            {
                return;
            }
            _startHeld = held;
            var scene = _world.GetComponent<MjScene>();
            if (scene != null)
            {
                scene.enabled = !held;
            }
        }

        /// <summary>
        /// Stops the world stepping, immediately, without destroying anything. Call this
        /// BEFORE the racers are destroyed.
        ///
        /// This exists because of a native crash, not for tidiness. Destroy() is deferred
        /// to end of frame, so the racers' MjComponents are still alive for the rest of
        /// the frame, and every one of their OnDisable/OnDestroy calls sets
        /// SceneRecreationAtLateUpdateRequested on MjScene. If MjScene's LateUpdate then
        /// runs before its own OnDestroy — and the order between two objects destroyed in
        /// the same frame is not defined — it calls RecreateScene() -> mj_resetData() on a
        /// model whose bodies are going away, and mujoco.dll takes the whole process down.
        /// The 2026-09-09 Editor crash was exactly that stack. Disabling the component
        /// first means no FixedUpdate and no LateUpdate, so neither the step nor the
        /// recreate can fire; OnDestroy still runs and still frees the native model.
        /// </summary>
        internal static void Suspend()
        {
            if (_world == null || _suspended)
            {
                return;
            }
            _suspended = true;
            var scene = _world.GetComponent<MjScene>();
            if (scene != null)
            {
                scene.enabled = false;
            }
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
            // Belt and braces: a caller that tore down without suspending first still gets
            // the stepping stopped before anything is queued for destruction.
            Suspend();
            Object.Destroy(_world);
            _world = null;
            _suspended = false;
            _startHeld = false;
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
        /// The contact properties every walkable MuJoCo surface shares, straight from
        /// creature.xml's &lt;geom&gt; default — which is what the policies trained against.
        /// The road must match the ground here or a racer's gait changes underfoot.
        /// </summary>
        private static void ApplyGroundFriction(MjGeom geom)
        {
            MjGeomSettings settings = geom.Settings;
            settings.Friction.Sliding = GROUND_FRICTION_SLIDING;
            settings.Friction.Torsional = GROUND_FRICTION_TORSIONAL;
            settings.Friction.Rolling = GROUND_FRICTION_ROLLING;
            settings.Solver.ConDim = GROUND_CONDIM;
            geom.Settings = settings;
        }

        /// <summary>
        /// Mirrors an authored course's centreline into MuJoCo as a chain of box geoms,
        /// one per centreline segment, so a MuJoCo racer has a road to stand on.
        ///
        /// WHY BOXES AND NOT THE COURSE'S OWN COLLIDERS. MuJoCo cannot see a Unity
        /// Collider at all, so the geometry has to be rebuilt on its side. Mirroring the
        /// GLB's COL_* proxies as MjMeshShape does not work: MuJoCo convex-hulls mesh
        /// geoms, and the course ships THREE proxies for the whole road, so the hulls
        /// would fill the valley solid and bury the track. MjHeightFieldShape is no good
        /// either - it needs a Unity Terrain, which this project does not use, and a
        /// heightfield cannot express a tunnel or stacked switchbacks anyway, because
        /// those are not a height function of (x, z).
        ///
        /// Boxes are exact primitives, cheap, native to MuJoCo, and the centreline the
        /// course already carries (55 knots on Acrobat, with a half-width) is all the
        /// data needed to lay them out. Only the floor is built - a racer has no use for
        /// a tunnel ceiling.
        ///
        /// Must run BEFORE MjScene compiles its model at the end of the frame, which is
        /// why it is called from <see cref="Build"/> rather than after the racers exist.
        /// </summary>
        private static void BuildCourseRoad(Transform parent, Views.RaceCourseView course)
        {
            Systems_CoursePath path = course.Path;
            if (path == null || path.Length <= 0f)
            {
                Debug.LogWarning("MuJoCo course road: the course has no usable centreline; "
                               + "MuJoCo racers will have no road.");
                return;
            }

            var road = new GameObject("MuJoCoCourseRoad");
            road.transform.SetParent(parent, false);

            // Sampled along the centreline rather than read off the knot array, so the
            // slab length is uniform and independent of how densely the course was
            // authored. One slab per ~2 m keeps a 26 deg bend inside half a slab.
            const float slabStep = 2f;
            int slabCount = Mathf.Max(1, Mathf.CeilToInt(path.Length / slabStep));
            float step = path.Length / slabCount;
            int built = 0;
            for (int slabIndex = 0; slabIndex < slabCount; slabIndex++)
            {
                float from = slabIndex * step;
                Vector3 a = path.PointAt(from);
                Vector3 b = path.PointAt(from + step);
                Vector3 along = b - a;
                if (along.sqrMagnitude < 0.0001f)
                {
                    continue;
                }

                var slab = new GameObject($"RoadSlab_{slabIndex:000}");
                slab.transform.SetParent(road.transform, false);
                // Centred on the segment and sunk by half its thickness, so the TOP face
                // sits on the centreline the racers' progress is measured along.
                slab.transform.position = (a + b) * 0.5f - Vector3.up * (ROAD_SLAB_THICKNESS * 0.5f);
                slab.transform.rotation = Quaternion.LookRotation(along.normalized, Vector3.up);

                var geom = slab.AddComponent<MjGeom>();
                geom.ShapeType = MjShapeComponent.ShapeTypes.Box;
                geom.Box.Extents = new Vector3(
                    course.HalfWidth,
                    ROAD_SLAB_THICKNESS * 0.5f,
                    along.magnitude * 0.5f * ROAD_SLAB_OVERLAP);
                ApplyGroundFriction(geom);
                built++;
            }
            Log($"MuJoCo course road: {built} slab(s) over {path.Length:0.0} m "
                    + $"at half-width {course.HalfWidth:0.0} m.");
        }

        /// <summary>
        /// Copies static colliders into the MuJoCo world as geoms of the same shape, pose
        /// and size. Only boxes and spheres: they map onto MuJoCo primitives exactly.
        /// Mesh colliders are skipped - MuJoCo convex-hulls meshes (see
        /// <see cref="BuildCourseRoad"/>), which would fill a course's valleys solid.
        /// Triggers and anything on a rigidbody or articulation are not scenery.
        /// Must run before MjScene compiles its model at the end of the frame.
        /// </summary>
        private static void MirrorSceneryColliders(Transform parent, Transform scenery)
        {
            var mirror = new GameObject("MuJoCoScenery");
            mirror.transform.SetParent(parent, false);
            Collider[] colliders = scenery.GetComponentsInChildren<Collider>();
            int mirrored = 0;
            for (int colliderIndex = 0; colliderIndex < colliders.Length; colliderIndex++)
            {
                Collider source = colliders[colliderIndex];
                if (!source.enabled || source.isTrigger
                    || source.attachedRigidbody != null || source.attachedArticulationBody != null)
                {
                    continue;
                }
                Transform sourceTransform = source.transform;
                Vector3 scale = sourceTransform.lossyScale;
                Vector3 center;
                MjGeom geom;
                if (source is BoxCollider box)
                {
                    center = box.center;
                    geom = AddGeom(mirror.transform, source.name, sourceTransform, center);
                    geom.ShapeType = MjShapeComponent.ShapeTypes.Box;
                    geom.Box.Extents = Vector3.Scale(box.size, Abs(scale)) * 0.5f;
                }
                else if (source is SphereCollider sphere)
                {
                    center = sphere.center;
                    geom = AddGeom(mirror.transform, source.name, sourceTransform, center);
                    geom.ShapeType = MjShapeComponent.ShapeTypes.Sphere;
                    geom.Sphere.Radius = sphere.radius * Mathf.Max(Mathf.Abs(scale.x), Mathf.Abs(scale.y), Mathf.Abs(scale.z));
                }
                else
                {
                    continue;
                }
                ApplyGroundFriction(geom);
                mirrored++;
            }
            Log($"MuJoCo scenery: {mirrored} collider(s) mirrored from '{scenery.name}'.");
        }

        private static MjGeom AddGeom(Transform parent, string name, Transform source, Vector3 localCenter)
        {
            var geomObject = new GameObject(name);
            geomObject.transform.SetParent(parent, false);
            geomObject.transform.SetPositionAndRotation(source.TransformPoint(localCenter), source.rotation);
            return geomObject.AddComponent<MjGeom>();
        }

        [System.Diagnostics.Conditional("UNITY_EDITOR")]
        private static void Log(string message) => Debug.Log(message);

        private static Vector3 Abs(Vector3 value) => new(Mathf.Abs(value.x), Mathf.Abs(value.y), Mathf.Abs(value.z));

        /// <summary>
        /// The MuJoCo ground. At y = 0 for a builder map — exactly where
        /// Systems_TrackBuilder puts the top of the flat track's collider slab, so a
        /// MuJoCo racer and the PhysX racers stand on the same surface without either
        /// engine knowing about the other. Pushed far down on a course, where the road
        /// above is the real surface and this is only a fall-catcher.
        /// </summary>
        private static void BuildGround(Transform parent, float height)
        {
            var ground = new GameObject("MuJoCoGround");
            ground.transform.SetParent(parent, false);
            ground.transform.localPosition = new Vector3(0f, height, 0f);
            ground.transform.localRotation = Quaternion.identity;

            var geom = ground.AddComponent<MjGeom>();
            geom.ShapeType = MjShapeComponent.ShapeTypes.Plane;
            geom.Plane.Extents = GroundExtents;
            ApplyGroundFriction(geom);

            // The track already draws a ground; this one only needs to exist for MuJoCo.
            // MjGeom's own mesh preview is added by the importer, not by AddComponent, so
            // there is nothing to hide here beyond leaving the renderer off.
        }
    }
}
