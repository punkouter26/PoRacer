#if UNITY_EDITOR
using System.Collections.Generic;
using PoRacer.Systems;
using PoRacer.Views;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace PoRacer.EditorTools
{
    /// <summary>
    /// Places the authored Apartment course (Assets/Art/Models/ApartmentTrack.glb)
    /// into SCN_RACE_FLAT as the AuthoredTrack_Apartment subtree and registers it on
    /// RaceTrackView, so the Apartment map races along the checkpoints in the GLB.
    ///
    /// The GLB is a photo reconstruction of a flat with a toy race track built
    /// through it, so its conventions are NOT the ones the Acrobat course uses:
    ///   Checkpoint_00..11        - the centreline knots -> RaceCourseView points
    ///   Race track road          - the drivable ribbon
    ///   Race track kerb/barrier/supports/centre dashes - the rest of the track body
    ///   Start finish line        - painted line at Checkpoint_00
    ///   Track centreline         - an empty Blender curve that exported with no
    ///                              geometry; the checkpoints are the usable path
    /// Everything else (2100-odd nodes of furniture, laundry and floorboards) is
    /// scenery: it keeps its renderer and gets NO collider, so racers pass through
    /// it and PhysX only ever sees the track. Putting MeshColliders on the whole
    /// reconstruction would cook 2139 meshes for nothing.
    ///
    /// THE LOOP IS DELIBERATELY LEFT OPEN. The checkpoints close back on
    /// Checkpoint_00, but Systems_CoursePath.Project is a global nearest-segment
    /// search with no monotonic guard, so on a closed ring a racer sitting on the
    /// start line is ambiguous between the first segment and the last and can be
    /// credited a whole lap at t=0. Racing knots 00..11 as an open path keeps the
    /// finish 1.4 m short of the start, which is clearance enough for the
    /// projection and costs only the last stretch of painted road.
    ///
    /// Re-runnable: replaces the previous subtree and entry.
    ///
    /// Invoke: unity command eval --code "return PoRacer.EditorTools.Editor_BuildApartmentTrack.Build();"
    /// </summary>
    public static class Editor_BuildApartmentTrack
    {
        private const string SCENE_PATH = "Assets/Scenes/SCN_RACE_FLAT.unity";
        private const string GLB_PATH = "Assets/Art/Models/ApartmentTrack.glb";
        private const string AUTHORED_NAME = "AuthoredTrack_Apartment";
        private const string CHECKPOINT_PREFIX = "Checkpoint_";
        private const string TRACK_PREFIX = "Race track";
        private const string START_LINE = "Start finish line";
        private const string FINISH_NAME = "Trigger_Finish";
        private const string SPAWN_NAME = "Spawn_01";

        /// <summary>
        /// Uniform scale the flat is placed at. The track is authored at toy size:
        /// a 0.64 m road bed on a 16.1 m lap, which no racer fits on.
        ///
        /// Racers cannot be scaled to suit it: their brains are trained against
        /// fixed torques, masses and gravity, so shrinking one breaks its policy.
        /// The map is the only side that moves - and because the model scales
        /// uniformly, road width and lap length move together and cannot be
        /// traded apart without re-authoring the GLB.
        ///
        /// Progress falls as the map grows, because a uniform scale preserves
        /// every GRADIENT while multiplying the LENGTH of each slope: the same
        /// 16 deg ramp is 8 m at scale 3 and 24 m at scale 9, and a brain trained
        /// on flat ground can carry the first but stalls partway up the last.
        /// That is the reason to keep this number small, and the reason the lap
        /// starts at the summit (see Install).
        ///
        /// 5.0 is a settlement, not an optimum: a 3.2 m road wide enough to read
        /// as a road the racers belong on, on a 73 m lap. Lower it if racing
        /// matters more than the look.
        ///
        /// Do NOT compare scales using progress figures measured before the
        /// two-segment gap landed (see Install). Until then Project could credit
        /// a racer most of a lap for drifting off the start line, and a 45 s race
        /// reported 59.4 m - 1.25 m/s, five times what the fastest brain does.
        /// The same race now reports 12.9 m, i.e. 0.25 m/s, which is the pace
        /// Systems_MapCatalog already documents. Any cross-scale table has to be
        /// re-measured on the fixed build.
        /// </summary>
        public const float APARTMENT_SCALE = 5.0f;

        // Measured off the model: outer kerb extent minus inner kerb extent is
        // 0.61 m on one side and 0.66 m on the other, so the ribbon is ~0.64 m
        // across and the half-width is half of that.
        private const float ROAD_HALF_WIDTH_MODEL = 0.32f;

        // Finish gate, in model units, so it scales with the instance: wide enough
        // to overhang the kerbs, tall enough that nothing hurdles it.
        private static readonly Vector3 FINISH_BOX_MODEL = new Vector3(1.6f, 1.2f, 0.4f);

        public static string Build()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                return "cannot build while in play mode - exit play mode and run this again";
            }
            var asset = AssetDatabase.LoadAssetAtPath<GameObject>(GLB_PATH);
            if (asset == null)
            {
                return "no GLB at " + GLB_PATH;
            }
            Scene scene = EditorSceneManager.OpenScene(SCENE_PATH, OpenSceneMode.Single);
            if (!scene.IsValid())
            {
                return "could not open " + SCENE_PATH;
            }
            RaceTrackView track = Object.FindFirstObjectByType<RaceTrackView>();
            if (track == null || track.TrackRoot == null)
            {
                return "no RaceTrackView with a TrackRoot in " + SCENE_PATH;
            }

            Transform previous = track.transform.Find(AUTHORED_NAME);
            if (previous != null)
            {
                Object.DestroyImmediate(previous.gameObject);
            }

            var instance = (GameObject)PrefabUtility.InstantiatePrefab(asset);
            instance.name = AUTHORED_NAME;
            instance.transform.SetParent(track.transform, worldPositionStays: false);
            instance.transform.localPosition = track.TrackRoot.localPosition;
            instance.transform.localRotation = track.TrackRoot.localRotation;
            instance.transform.localScale = track.TrackRoot.localScale * APARTMENT_SCALE;

            string installError = Install(instance, track.PhysicsMaterial, out RaceCourseView course, out string summary);
            if (installError != null)
            {
                Object.DestroyImmediate(instance);
                return installError;
            }

            // Register on RaceTrackView: replace any earlier Apartment entry, keep the rest.
            var entries = new List<RaceTrackView.AuthoredTrack>();
            IReadOnlyList<RaceTrackView.AuthoredTrack> existing = track.AuthoredTracks;
            for (int existingIndex = 0; existingIndex < existing.Count; existingIndex++)
            {
                if (existing[existingIndex].kind != TrackKind.Apartment)
                {
                    entries.Add(existing[existingIndex]);
                }
            }
            entries.Add(new RaceTrackView.AuthoredTrack
            {
                root = instance.transform,
                kind = TrackKind.Apartment,
                lengthMeters = course.Path.Length,
                features = TrackFeatures.None,
                groundBounds = course.Bounds,
                hasArch = false,
                archBounds = default,
                course = course,
            });
            var trackSerialized = new SerializedObject(track);
            SerializedProperty list = trackSerialized.FindProperty("_authoredTracks");
            list.arraySize = entries.Count;
            for (int entryIndex = 0; entryIndex < entries.Count; entryIndex++)
            {
                SerializedProperty element = list.GetArrayElementAtIndex(entryIndex);
                RaceTrackView.AuthoredTrack entry = entries[entryIndex];
                element.FindPropertyRelative("root").objectReferenceValue = entry.root;
                element.FindPropertyRelative("kind").enumValueIndex = (int)entry.kind;
                element.FindPropertyRelative("lengthMeters").floatValue = entry.lengthMeters;
                element.FindPropertyRelative("features").intValue = (int)entry.features;
                element.FindPropertyRelative("groundBounds").boundsValue = entry.groundBounds;
                element.FindPropertyRelative("hasArch").boolValue = entry.hasArch;
                element.FindPropertyRelative("archBounds").boundsValue = entry.archBounds;
                element.FindPropertyRelative("course").objectReferenceValue = entry.course;
            }
            trackSerialized.ApplyModifiedPropertiesWithoutUndo();

            // Hidden until the Apartment map is picked; Systems_Spawn shows it.
            instance.SetActive(false);
            EditorSceneManager.MarkSceneDirty(scene);
            if (!EditorSceneManager.SaveScene(scene))
            {
                return "SaveScene failed for " + SCENE_PATH;
            }
            return AUTHORED_NAME + ": " + summary;
        }

        /// <summary>
        /// Turns a freshly instantiated copy of the GLB into a raceable course:
        /// colliders on the track body only, a start marker and finish gate built
        /// from the checkpoints, and a RaceCourseView carrying the centreline,
        /// spawn and bounds. Returns an error string or null.
        /// </summary>
        public static string Install(GameObject instance, PhysicsMaterial physicsMaterial,
            out RaceCourseView course, out string summary)
        {
            course = null;
            summary = string.Empty;
            var knots = new List<(int index, Vector3 world)>();
            Bounds bounds = default;
            bool hasBounds = false;
            int colliders = 0;

            Transform[] all = instance.GetComponentsInChildren<Transform>(true);
            for (int index = 0; index < all.Length; index++)
            {
                Transform node = all[index];
                string name = node.name;

                if (name.StartsWith(CHECKPOINT_PREFIX)
                    && int.TryParse(name.Substring(CHECKPOINT_PREFIX.Length), out int knot))
                {
                    knots.Add((knot, instance.transform.TransformPoint(node.localPosition)));
                    continue;
                }

                // The track body is the only thing PhysX may see. Double-sided for
                // the same reason the Acrobat proxies are: a ribbon exported from
                // Blender can carry either winding and a racer must not fall through.
                if (name.StartsWith(TRACK_PREFIX) || name == START_LINE)
                {
                    MeshFilter filter = node.GetComponent<MeshFilter>();
                    if (filter != null && filter.sharedMesh != null)
                    {
                        var meshCollider = node.gameObject.AddComponent<MeshCollider>();
                        meshCollider.sharedMesh = Editor_BuildCourseTrack.DoubleSided(filter.sharedMesh);
                        meshCollider.sharedMaterial = physicsMaterial;
                        colliders++;
                    }
                    Renderer trackRenderer = node.GetComponent<Renderer>();
                    if (trackRenderer != null)
                    {
                        if (!hasBounds)
                        {
                            bounds = trackRenderer.bounds;
                            hasBounds = true;
                        }
                        else
                        {
                            bounds.Encapsulate(trackRenderer.bounds);
                        }
                    }
                }
            }

            if (knots.Count < 2)
            {
                return "GLB carries no " + CHECKPOINT_PREFIX + "* knots; nothing to race along";
            }
            if (!hasBounds)
            {
                return "GLB carries no '" + TRACK_PREFIX + "*' geometry; nothing to stand on";
            }

            knots.Sort((first, second) => first.index.CompareTo(second.index));

            // Start the lap at the highest knot, not at Checkpoint_00.
            //
            // The toy track ramps straight off its painted line onto the raised
            // section: measured at APARTMENT_SCALE 9 that was 6.6 m of climb at
            // ~16 deg over the first 24 m, and every racer stalled partway up,
            // slid back, and was DNF'd by the 30 s no-progress rule without ever
            // clearing it. Scaling makes this worse rather than better, because a
            // uniform scale preserves the GRADIENT but multiplies the LENGTH of
            // the climb - at scale 3 the same ramp was only 8 m long and the pack
            // ran straight over it.
            //
            // The checkpoints are a closed ring, so the lap can begin anywhere on
            // it. Beginning at the summit puts the long descent and flat first and
            // leaves the climb as the last stretch, where only a leader reaches it.
            int summitIndex = 0;
            for (int knotIndex = 1; knotIndex < knots.Count; knotIndex++)
            {
                if (knots[knotIndex].world.y > knots[summitIndex].world.y)
                {
                    summitIndex = knotIndex;
                }
            }
            // Drop the last knot, so the ring is left open by TWO segments rather
            // than one. Systems_CoursePath.Project searches every segment and
            // returns the nearest, with no monotonic guard, so a racer that drifts
            // back off the start line is credited the whole lap the moment it is
            // marginally closer to the final segment than to the first. One
            // segment of clearance (7.2 m at scale 5) was not enough: a 45 s race
            // reported racers at 48 m and 59 m, which is over 1.2 m/s for brains
            // that walk at about a quarter of that. Two segments roughly doubles
            // the margin, at the cost of one knot's worth of lap.
            int pointCount = knots.Count - 1;
            var points = new Vector3[pointCount];
            for (int knotIndex = 0; knotIndex < pointCount; knotIndex++)
            {
                points[knotIndex] = knots[(summitIndex + knotIndex) % knots.Count].world;
            }

            // Start marker at the first knot, facing the second: the spawner fans
            // the rest of the pack back along the centreline from here.
            var spawnMarker = new GameObject(SPAWN_NAME);
            spawnMarker.transform.SetParent(instance.transform, worldPositionStays: false);
            spawnMarker.transform.position = points[0];
            spawnMarker.transform.rotation = Quaternion.LookRotation(
                Vector3.Normalize(points[1] - points[0]), Vector3.up);

            // Finish gate across the last knot, square to the run in.
            var finish = new GameObject(FINISH_NAME);
            finish.transform.SetParent(instance.transform, worldPositionStays: false);
            finish.transform.position = points[points.Length - 1];
            finish.transform.rotation = Quaternion.LookRotation(
                Vector3.Normalize(points[points.Length - 1] - points[points.Length - 2]), Vector3.up);
            var finishBox = finish.AddComponent<BoxCollider>();
            finishBox.isTrigger = true;
            finishBox.size = FINISH_BOX_MODEL;
            finish.AddComponent<CourseFinishView>();

            course = instance.AddComponent<RaceCourseView>();
            var courseSerialized = new SerializedObject(course);
            SerializedProperty pointsProperty = courseSerialized.FindProperty("_points");
            pointsProperty.arraySize = points.Length;
            for (int pointIndex = 0; pointIndex < points.Length; pointIndex++)
            {
                pointsProperty.GetArrayElementAtIndex(pointIndex).vector3Value = points[pointIndex];
            }
            SerializedProperty spawnsProperty = courseSerialized.FindProperty("_spawnPoints");
            spawnsProperty.arraySize = 1;
            spawnsProperty.GetArrayElementAtIndex(0).objectReferenceValue = spawnMarker.transform;
            courseSerialized.FindProperty("_bounds").boundsValue = bounds;
            courseSerialized.FindProperty("_halfWidth").floatValue =
                ROAD_HALF_WIDTH_MODEL * instance.transform.lossyScale.x;
            courseSerialized.ApplyModifiedPropertiesWithoutUndo();

            var path = new Systems_CoursePath(points);
            summary = $"{points.Length} knots, {path.Length:0.0} m, {colliders} track colliders, " +
                $"half-width {ROAD_HALF_WIDTH_MODEL * instance.transform.lossyScale.x:0.00} m, " +
                $"bounds {bounds.size:F1} at {bounds.center:F1}; start {path.Start:F1} finish {path.End:F1}";
            return null;
        }
    }
}
#endif
