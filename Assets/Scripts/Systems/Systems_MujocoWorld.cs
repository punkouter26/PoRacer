using System.Collections.Generic;
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

        // Fruit stand-ins (see TrackFruit): built with the world, borrowed per piece.
        private static readonly float[] FruitProxyRadii = { 0.15f, 0.25f, 0.35f };
        private const int FRUIT_PROXIES_PER_SIZE = 100;
        private const float FRUIT_SLIDING_FRICTION = 0.3f;
        private static readonly Vector3 FruitParking = new(0f, -200f, 0f);
        private static readonly List<Views.MujocoFruitProxyView> FruitProxies = new();
        private static bool _suspended;
        private static bool _startHeld;
        private static Views.RaceCourseView _pendingCourse;

        // The course, mirrored into MuJoCo by probing the course's own PhysX colliders
        // (see BuildCourseRoad): floor strips under the whole road width, walls where the
        // barriers and tunnel walls are.
        private const float ROAD_STEP = 1.5f;             // metres of centreline per row of strips
        private const float ROAD_STRIP_WIDTH = 0.8f;      // metres across per floor strip
        private const float ROAD_FLOOR_MARGIN = 1.5f;     // probed beyond the half-width: kerbs, verges
        private const float ROAD_SLAB_THICKNESS = 0.3f;   // depth under the surface; pushes a sunk foot out
        private const float ROAD_PROBE_UP = 1.2f;         // down-rays start this far above the centreline
        private const float ROAD_PROBE_DOWN = 3f;
        private const float ROAD_MAX_STEP = 0.6f;         // a hit further than this from the centreline height is another level
        // Sideways rays find barriers: first just above the road, then stepping up to find
        // each barrier's real top, so the wall is as tall as the barrier a PhysX racer
        // meets and no taller. The Apartment's barriers are only 0.2-0.3 m, and a single
        // ray at knee height (0.35 m) flew over every one of them.
        private const float WALL_PROBE_HEIGHT = 0.05f;
        private const float WALL_PROBE_STEP = 0.05f;
        private const float WALL_MAX_HEIGHT = 2f;
        private const float WALL_PROBE_REACH = 3f;        // beyond the half-width
        private const float WALL_THICKNESS = 0.25f;
        private const float WALL_MAX_NORMAL_Y = 0.5f;     // steeper than 60 degrees is a wall, not floor

        /// <summary>
        /// Every strip and wall is stretched this much past its segment. Butted end to end
        /// they leave a wedge of gap on the outside of every bend; overlapping costs
        /// nothing, because MuJoCo static geoms do not collide with each other.
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
            BuildFruitProxies(_world.transform);

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
            FruitProxies.Clear();
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
        /// Mirrors an authored course into MuJoCo as box geoms, measured off the course's
        /// OWN PhysX colliders, so a MuJoCo racer stands, and is stopped, exactly where a
        /// PhysX racer is.
        ///
        /// WHY NOT THE COURSE'S MESH COLLIDERS DIRECTLY. MuJoCo cannot see a Unity Collider,
        /// and it convex-hulls mesh geoms: one hull per road, kerb or barrier mesh would
        /// fill the course solid. MjHeightFieldShape needs a Unity Terrain, and a heightfield
        /// cannot express a tunnel or stacked switchbacks anyway.
        ///
        /// WHY PROBED, NOT LAID ALONG THE CENTRELINE. The first version was one 2 m slab per
        /// centreline segment, as wide as the course's half-width and at the centreline's
        /// height. Measured on the Apartment track: the road plus kerbs is wider than that,
        /// so a racer near the edge stood half on the slab and half over nothing and hung
        /// there, sunk to the hips; the kerb and barrier meshes were not mirrored at all, so
        /// MuJoCo racers walked through them; and the centreline sits up to 0.21 m off the
        /// real surface. Now, every ROAD_STEP metres:
        ///
        ///   * down-rays across the half-width plus ROAD_FLOOR_MARGIN give the surface
        ///     height strip by strip, kerbs included, and a floor strip is laid on it;
        ///   * sideways rays at knee height find the barriers and tunnel walls either side,
        ///     and a wall box is stood where they are.
        ///
        /// Only static colliders count (never a racer), and a down-ray hit far from the
        /// centreline's height is ignored as another level of the course.
        /// Must run BEFORE MjScene compiles its model at the end of the frame.
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
            // The track was just switched on by the spawner; make sure PhysX sees it.
            Physics.SyncTransforms();

            var road = new GameObject("MuJoCoCourseRoad");
            road.transform.SetParent(parent, false);

            float reach = course.HalfWidth + ROAD_FLOOR_MARGIN;
            int strips = Mathf.Max(1, Mathf.CeilToInt(2f * reach / ROAD_STRIP_WIDTH));
            float stripWidth = 2f * reach / strips;
            int rows = Mathf.Max(1, Mathf.CeilToInt(path.Length / ROAD_STEP));
            float step = path.Length / rows;
            var hits = new RaycastHit[16];
            int floors = 0;
            int walls = 0;
            for (int row = 0; row < rows; row++)
            {
                Vector3 a = path.PointAt(row * step);
                Vector3 b = path.PointAt((row + 1) * step);
                Vector3 along = b - a;
                along.y = 0f;
                if (along.sqrMagnitude < 0.0001f)
                {
                    continue;
                }
                along.Normalize();
                Vector3 right = Vector3.Cross(Vector3.up, along);

                for (int strip = 0; strip < strips; strip++)
                {
                    float offset = -reach + (strip + 0.5f) * stripWidth;
                    if (!ProbeFloor(a + right * offset, a.y, hits, out Vector3 start)
                        || !ProbeFloor(b + right * offset, b.y, hits, out Vector3 end))
                    {
                        continue;
                    }
                    Vector3 run = end - start;
                    if (run.sqrMagnitude < 0.0001f)
                    {
                        continue;
                    }
                    Quaternion rotation = Quaternion.LookRotation(run.normalized, Vector3.up);
                    Vector3 centre = (start + end) * 0.5f - rotation * Vector3.up * (ROAD_SLAB_THICKNESS * 0.5f);
                    AddBox(road.transform, "Floor_" + row + "_" + strip, centre, rotation,
                        new Vector3(stripWidth * 0.5f * ROAD_SLAB_OVERLAP, ROAD_SLAB_THICKNESS * 0.5f,
                                    run.magnitude * 0.5f * ROAD_SLAB_OVERLAP));
                    floors++;
                }

                Vector3 middle = (a + b) * 0.5f;
                if (!ProbeFloor(middle, middle.y, hits, out Vector3 surface))
                {
                    surface = middle;
                }
                for (int side = -1; side <= 1; side += 2)
                {
                    Vector3 direction = right * side;
                    if (!ProbeWall(surface + Vector3.up * WALL_PROBE_HEIGHT, direction,
                                   course.HalfWidth + WALL_PROBE_REACH, hits, out Vector3 wallPoint))
                    {
                        continue;
                    }
                    // Step up until the rays pass over it: that is the barrier's top.
                    float reachToWall = Vector3.Distance(surface + Vector3.up * WALL_PROBE_HEIGHT, wallPoint) + WALL_THICKNESS;
                    float height = WALL_PROBE_HEIGHT;
                    while (height < WALL_MAX_HEIGHT
                           && ProbeWall(surface + Vector3.up * (height + WALL_PROBE_STEP), direction,
                                        reachToWall, hits, out _))
                    {
                        height += WALL_PROBE_STEP;
                    }
                    height += WALL_PROBE_STEP * 0.5f;
                    Vector3 centre = new Vector3(wallPoint.x, surface.y, wallPoint.z)
                                   + direction * (WALL_THICKNESS * 0.5f) + Vector3.up * (height * 0.5f);
                    AddBox(road.transform, "Wall_" + row + (side < 0 ? "_L" : "_R"), centre,
                        Quaternion.LookRotation(along, Vector3.up),
                        new Vector3(WALL_THICKNESS * 0.5f, height * 0.5f, step * 0.5f * ROAD_SLAB_OVERLAP));
                    walls++;
                }
            }
            Log($"MuJoCo course road: {floors} floor strip(s), {walls} wall(s) over {path.Length:0.0} m "
              + $"({rows} rows x {strips} strips of {stripWidth:0.00} m).");
        }

        /// <summary>The course surface under <paramref name="point"/>: the static hit nearest
        /// the centreline's height, so a tunnel roof or a lower switchback is not taken.</summary>
        private static bool ProbeFloor(Vector3 point, float centreHeight, RaycastHit[] hits, out Vector3 surface)
        {
            Vector3 origin = new(point.x, centreHeight + ROAD_PROBE_UP, point.z);
            int count = Physics.RaycastNonAlloc(origin, Vector3.down, hits, ROAD_PROBE_UP + ROAD_PROBE_DOWN,
                                                Physics.DefaultRaycastLayers, QueryTriggerInteraction.Ignore);
            surface = default;
            float best = float.PositiveInfinity;
            for (int index = 0; index < count; index++)
            {
                RaycastHit hit = hits[index];
                if (!IsStatic(hit.collider) || hit.normal.y <= WALL_MAX_NORMAL_Y)
                {
                    continue;
                }
                float distance = Mathf.Abs(hit.point.y - centreHeight);
                if (distance <= ROAD_MAX_STEP && distance < best)
                {
                    best = distance;
                    surface = hit.point;
                }
            }
            return best < float.PositiveInfinity;
        }

        /// <summary>The nearest steep static surface along <paramref name="direction"/>.</summary>
        private static bool ProbeWall(Vector3 origin, Vector3 direction, float reach, RaycastHit[] hits, out Vector3 point)
        {
            int count = Physics.RaycastNonAlloc(origin, direction, hits, reach,
                                                Physics.DefaultRaycastLayers, QueryTriggerInteraction.Ignore);
            point = default;
            float best = float.PositiveInfinity;
            for (int index = 0; index < count; index++)
            {
                RaycastHit hit = hits[index];
                if (IsStatic(hit.collider) && Mathf.Abs(hit.normal.y) < WALL_MAX_NORMAL_Y && hit.distance < best)
                {
                    best = hit.distance;
                    point = hit.point;
                }
            }
            return best < float.PositiveInfinity;
        }

        private static bool IsStatic(Collider collider) =>
            collider.attachedRigidbody == null && collider.attachedArticulationBody == null;

        private static void AddBox(Transform parent, string name, Vector3 position, Quaternion rotation, Vector3 extents)
        {
            var box = new GameObject(name);
            box.transform.SetParent(parent, false);
            box.transform.SetPositionAndRotation(position, rotation);
            var geom = box.AddComponent<MjGeom>();
            geom.ShapeType = MjShapeComponent.ShapeTypes.Box;
            geom.Box.Extents = extents;
            ApplyGroundFriction(geom);
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
        /// Lets MuJoCo racers bump into a piece of fruit (AGENTS rule M): the free pooled
        /// stand-in nearest its size follows it until the piece is destroyed. A no-op when
        /// no MuJoCo racer is on the grid. The fruit stays a PhysX rigidbody, so every PhysX
        /// racer collides with it natively; MuJoCo racers reach it through this, and it
        /// reaches them through their MujocoPhysxProxyView.
        /// </summary>
        internal static void TrackFruit(Transform piece, float radius)
        {
            if (!Exists)
            {
                return;
            }
            Views.MujocoFruitProxyView best = null;
            float bestGap = float.PositiveInfinity;
            for (int proxyIndex = 0; proxyIndex < FruitProxies.Count; proxyIndex++)
            {
                Views.MujocoFruitProxyView proxy = FruitProxies[proxyIndex];
                float gap = Mathf.Abs(proxy.Radius - radius);
                if (proxy != null && proxy.IsFree && gap < bestGap)
                {
                    best = proxy;
                    bestGap = gap;
                }
            }
            if (best != null)
            {
                best.Bind(piece);
            }
        }

        /// <summary>
        /// The fruit stand-in pool: mocap spheres parked far below the world. Built with the
        /// world because a body added after the model compiles makes the plug-in recreate
        /// the scene, which resets every MuJoCo racer. Parked stand-ins make no contacts:
        /// MuJoCo skips pairs of static bodies, and a mocap body counts as one.
        /// </summary>
        private static void BuildFruitProxies(Transform parent)
        {
            FruitProxies.Clear();
            var pool = new GameObject("MuJoCoFruitProxies");
            pool.transform.SetParent(parent, false);
            for (int sizeIndex = 0; sizeIndex < FruitProxyRadii.Length; sizeIndex++)
            {
                float radius = FruitProxyRadii[sizeIndex];
                for (int proxyIndex = 0; proxyIndex < FRUIT_PROXIES_PER_SIZE; proxyIndex++)
                {
                    var body = new GameObject($"Fruit_{sizeIndex}_{proxyIndex:000}");
                    body.transform.SetParent(pool.transform, false);
                    body.transform.position = FruitParking;
                    body.AddComponent<MjMocapBody>();
                    var geomObject = new GameObject("geom");
                    geomObject.transform.SetParent(body.transform, false);
                    MjGeom geom = geomObject.AddComponent<MjGeom>();
                    geom.ShapeType = MjShapeComponent.ShapeTypes.Sphere;
                    geom.Sphere.Radius = radius;
                    MjGeomSettings settings = geom.Settings;
                    settings.Friction.Sliding = FRUIT_SLIDING_FRICTION;
                    geom.Settings = settings;
                    var view = body.AddComponent<Views.MujocoFruitProxyView>();
                    view.Initialize(radius, FruitParking);
                    FruitProxies.Add(view);
                }
            }
        }

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
