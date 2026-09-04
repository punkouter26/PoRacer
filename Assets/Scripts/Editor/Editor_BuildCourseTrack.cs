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
    /// Places the authored Acrobat course (Assets/Art/Models/AcrobatTrack.glb) into
    /// SCN_RACE_FLAT as the AuthoredTrack_Acrobat subtree and registers it on
    /// RaceTrackView, so the Acrobat map races along the GLB's own centreline.
    ///
    /// What the GLB carries and what this reads off it:
    ///   Body/Body_LOD0..2          - the mountain, three detail levels -> one LODGroup
    ///   Colliders/COL_*            - collision proxies -> MeshColliders, renderers off
    ///   Track/SplinePoints/*       - the centreline knots -> RaceCourseView points
    ///   Race/Spawn_01..04          - start markers -> RaceCourseView spawn points
    ///   Race/Trigger_Finish        - the finish volume -> BoxCollider trigger + CourseFinishView
    ///   Track/TunnelLight_*        - lamp positions -> point lights
    ///
    /// The spline knots are read as MODEL-SPACE positions. Blender exported the
    /// SplinePoints empty with its own translation while the knots kept world
    /// coordinates, so reading them through that parent lands the course 40 m
    /// off; the knots' local values line up with the spawn and trigger markers.
    ///
    /// Re-runnable: replaces the previous subtree and entry. Course entries are
    /// left alone by Editor_BakeAuthoredTrack.
    ///
    /// Invoke: unity command eval --code "return PoRacer.EditorTools.Editor_BuildCourseTrack.Build();"
    /// </summary>
    public static class Editor_BuildCourseTrack
    {
        private const string SCENE_PATH = "Assets/Scenes/SCN_RACE_FLAT.unity";
        private const string GLB_PATH = "Assets/Art/Models/AcrobatTrack.glb";
        private const string AUTHORED_NAME = "AuthoredTrack_Acrobat";
        private const string COLLIDER_PREFIX = "COL_";
        private const string SPLINE_PREFIX = "SplinePt_";
        private const string SPAWN_PREFIX = "Spawn_";
        private const string FINISH_TRIGGER = "Trigger_Finish";
        private const string TUNNEL_LIGHT_PREFIX = "TunnelLight_";
        private const string ROAD_SURFACE = "RaceTrack_Surface";
        private const string TUNNEL_LINING = "Tunnel_Lining";
        /// <summary>
        /// Uniform scale the course is placed at, so every grade and camber is
        /// preserved; everything below is read off the scaled instance. 1.0 is the
        /// authored size: a 5 m road bed (three racers abreast), 212 m of
        /// switchbacks, 50 m of climb. It was tried at 0.5 first and the pack had no
        /// room on the road; raise it before lowering it.
        /// </summary>
        public const float COURSE_SCALE = 1.0f;
        // Measured on the model: the road bed is about 5 m across before the
        // verge drops away (the 7 m trigger volumes overhang it).
        private const float ROAD_HALF_WIDTH_MODEL = 2.5f;
        private const float TUNNEL_LIGHT_RANGE = 14f;
        private const float TUNNEL_LIGHT_INTENSITY = 2.5f;

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
            instance.transform.localScale = track.TrackRoot.localScale * COURSE_SCALE;

            string installError = Install(instance, track.PhysicsMaterial, out RaceCourseView course, out string summary);
            if (installError != null)
            {
                Object.DestroyImmediate(instance);
                return installError;
            }
            Bounds bounds = course.Bounds;
            var path = course.Path;

            // Register on RaceTrackView: replace any earlier Course entry, keep the rest.
            var entries = new List<RaceTrackView.AuthoredTrack>();
            IReadOnlyList<RaceTrackView.AuthoredTrack> existing = track.AuthoredTracks;
            for (int existingIndex = 0; existingIndex < existing.Count; existingIndex++)
            {
                if (existing[existingIndex].kind != TrackKind.Course)
                {
                    entries.Add(existing[existingIndex]);
                }
            }
            entries.Add(new RaceTrackView.AuthoredTrack
            {
                root = instance.transform,
                kind = TrackKind.Course,
                lengthMeters = path.Length,
                features = TrackFeatures.None,
                groundBounds = bounds,
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

            // Hidden until the Acrobat map is picked; Systems_Spawn shows it.
            instance.SetActive(false);
            EditorSceneManager.MarkSceneDirty(scene);
            if (!EditorSceneManager.SaveScene(scene))
            {
                return "SaveScene failed for " + SCENE_PATH;
            }
            return AUTHORED_NAME + ": " + summary;
        }

        /// <summary>The imported GLB, ready to instantiate; null when it is missing.</summary>
        public static GameObject LoadAsset() => AssetDatabase.LoadAssetAtPath<GameObject>(GLB_PATH);

        /// <summary>
        /// A collision copy of <paramref name="source"/> with every triangle present in
        /// both windings, so PhysX contacts and raycasts work from either side. Lives
        /// in the scene file with the collider that references it.
        /// </summary>
        private static Mesh DoubleSided(Mesh source)
        {
            int[] triangles = source.triangles;
            var both = new int[triangles.Length * 2];
            System.Array.Copy(triangles, both, triangles.Length);
            for (int triangleIndex = 0; triangleIndex < triangles.Length; triangleIndex += 3)
            {
                both[triangles.Length + triangleIndex] = triangles[triangleIndex];
                both[triangles.Length + triangleIndex + 1] = triangles[triangleIndex + 2];
                both[triangles.Length + triangleIndex + 2] = triangles[triangleIndex + 1];
            }
            var mesh = new Mesh
            {
                name = source.name + "_Collision",
                indexFormat = both.Length > 65000
                    ? UnityEngine.Rendering.IndexFormat.UInt32
                    : UnityEngine.Rendering.IndexFormat.UInt16,
                vertices = source.vertices,
                triangles = both,
            };
            mesh.RecalculateBounds();
            return mesh;
        }

        /// <summary>
        /// Turns a freshly instantiated copy of the GLB into a raceable course:
        /// colliders, LODs, lamps, finish trigger, and a RaceCourseView carrying the
        /// centreline, spawns and bounds. Returns an error string or null.
        /// </summary>
        public static string Install(GameObject instance, PhysicsMaterial physicsMaterial,
            out RaceCourseView course, out string summary)
        {
            course = null;
            summary = string.Empty;
            var splineKnots = new List<(int index, Vector3 world)>();
            var spawns = new List<(int index, Transform marker)>();
            var lodRenderers = new List<Renderer>();
            Transform finishTrigger = null;
            Bounds bounds = default;
            bool hasBounds = false;
            int colliders = 0;
            int lights = 0;

            Transform[] all = instance.GetComponentsInChildren<Transform>(true);
            for (int index = 0; index < all.Length; index++)
            {
                Transform node = all[index];
                string name = node.name;
                if (name.StartsWith(COLLIDER_PREFIX) || name == ROAD_SURFACE || name == TUNNEL_LINING)
                {
                    // Every collision proxy in the GLB is wound inside-out (COL_Track has
                    // no upward-facing triangle at all) and the road proxy stops short
                    // of the start straight, so racers fell through to the terrain body
                    // beneath the road and stood half-sunk in it. Cook each mesh
                    // double-sided so winding cannot matter, and collide on the visible
                    // road and tunnel surfaces themselves so racers rest exactly where
                    // the road is drawn.
                    MeshFilter filter = node.GetComponent<MeshFilter>();
                    if (filter != null && filter.sharedMesh != null)
                    {
                        var meshCollider = node.gameObject.AddComponent<MeshCollider>();
                        meshCollider.sharedMesh = DoubleSided(filter.sharedMesh);
                        meshCollider.sharedMaterial = physicsMaterial;
                        if (name.StartsWith(COLLIDER_PREFIX))
                        {
                            Renderer proxyRenderer = node.GetComponent<Renderer>();
                            if (proxyRenderer != null)
                            {
                                proxyRenderer.enabled = false;
                            }
                        }
                        colliders++;
                    }
                }
                else if (name.StartsWith(SPLINE_PREFIX) && int.TryParse(name.Substring(SPLINE_PREFIX.Length), out int knot))
                {
                    // Model-space knot, deliberately not node.position (see the class summary).
                    splineKnots.Add((knot, instance.transform.TransformPoint(node.localPosition)));
                }
                else if (name.StartsWith(SPAWN_PREFIX) && int.TryParse(name.Substring(SPAWN_PREFIX.Length), out int spawn))
                {
                    spawns.Add((spawn, node));
                }
                else if (name == FINISH_TRIGGER)
                {
                    finishTrigger = node;
                }
                else if (name.StartsWith(TUNNEL_LIGHT_PREFIX))
                {
                    var lamp = node.gameObject.AddComponent<Light>();
                    lamp.type = LightType.Point;
                    lamp.range = TUNNEL_LIGHT_RANGE;
                    lamp.intensity = TUNNEL_LIGHT_INTENSITY;
                    lamp.color = new Color(1f, 0.87f, 0.7f);
                    lights++;
                }
                else if (name.StartsWith("Body_LOD"))
                {
                    Renderer lodRenderer = node.GetComponent<Renderer>();
                    if (lodRenderer != null)
                    {
                        lodRenderers.Add(lodRenderer);
                    }
                }

                Renderer renderer = node.GetComponent<Renderer>();
                if (renderer != null && renderer.enabled && !name.StartsWith("Body_LOD"))
                {
                    if (!hasBounds)
                    {
                        bounds = renderer.bounds;
                        hasBounds = true;
                    }
                    else
                    {
                        bounds.Encapsulate(renderer.bounds);
                    }
                }
            }
            if (lodRenderers.Count > 0)
            {
                bounds.Encapsulate(lodRenderers[0].bounds);
            }

            if (splineKnots.Count < 2)
            {
                return "GLB carries no SplinePt_* knots; nothing to race along";
            }
            if (finishTrigger == null)
            {
                return "GLB carries no " + FINISH_TRIGGER;
            }

            splineKnots.Sort((first, second) => first.index.CompareTo(second.index));
            spawns.Sort((first, second) => first.index.CompareTo(second.index));
            var points = new Vector3[splineKnots.Count];
            for (int knotIndex = 0; knotIndex < points.Length; knotIndex++)
            {
                points[knotIndex] = splineKnots[knotIndex].world;
            }
            var spawnMarkers = new Transform[spawns.Count];
            for (int spawnIndex = 0; spawnIndex < spawnMarkers.Length; spawnIndex++)
            {
                spawnMarkers[spawnIndex] = spawns[spawnIndex].marker;
            }

            // LODs: the three mountain meshes share one LODGroup instead of all
            // drawing at once.
            if (lodRenderers.Count == 3)
            {
                var lodGroup = instance.AddComponent<LODGroup>();
                lodGroup.SetLODs(new[]
                {
                    new LOD(0.35f, new[] { lodRenderers[0] }),
                    new LOD(0.12f, new[] { lodRenderers[1] }),
                    new LOD(0.02f, new[] { lodRenderers[2] }),
                });
                lodGroup.RecalculateBounds();
            }

            // Finish volume: the authored empty carries the size in its scale, so a
            // unit box trigger fills exactly the gate.
            var finishBox = finishTrigger.gameObject.AddComponent<BoxCollider>();
            finishBox.isTrigger = true;
            finishBox.size = Vector3.one;
            finishTrigger.gameObject.AddComponent<CourseFinishView>();

            course = instance.AddComponent<RaceCourseView>();
            var courseSerialized = new SerializedObject(course);
            SerializedProperty pointsProperty = courseSerialized.FindProperty("_points");
            pointsProperty.arraySize = points.Length;
            for (int pointIndex = 0; pointIndex < points.Length; pointIndex++)
            {
                pointsProperty.GetArrayElementAtIndex(pointIndex).vector3Value = points[pointIndex];
            }
            SerializedProperty spawnsProperty = courseSerialized.FindProperty("_spawnPoints");
            spawnsProperty.arraySize = spawnMarkers.Length;
            for (int spawnIndex = 0; spawnIndex < spawnMarkers.Length; spawnIndex++)
            {
                spawnsProperty.GetArrayElementAtIndex(spawnIndex).objectReferenceValue = spawnMarkers[spawnIndex];
            }
            courseSerialized.FindProperty("_bounds").boundsValue = bounds;
            courseSerialized.FindProperty("_halfWidth").floatValue = ROAD_HALF_WIDTH_MODEL * instance.transform.lossyScale.x;
            courseSerialized.ApplyModifiedPropertiesWithoutUndo();

            var path = new Systems_CoursePath(points);
            summary = $"{points.Length} knots, {path.Length:0.0} m, {spawnMarkers.Length} spawns, " +
                $"{colliders} colliders, {lights} lamps, bounds {bounds.size:F0} at {bounds.center:F0}; " +
                $"start {path.Start:F1} finish {path.End:F1}";
            return null;
        }
    }
}
#endif
