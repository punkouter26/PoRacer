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
    /// Bakes every deterministic map into SCN_RACE_FLAT as ordinary scene objects,
    /// once, so each can be tuned by hand instead of being regenerated on every race.
    ///
    /// One subtree per map (AuthoredTrack_Flat, _Lumpy, _Swamp, _Gale), all listed on
    /// RaceTrackView. Systems_Spawn shows the one matching the selected map and hides
    /// the rest; only Roulette still goes through Systems_TrackBuilder at runtime,
    /// because it re-rolls terrain and hazards every race by design.
    ///
    /// Determinism. The generated maps varied only in decoration and hazard placement,
    /// both drawn from the per-race RNG. This bakes one roll per map from a fixed seed
    /// and keeps it; after that the scene is the source of truth and re-running this
    /// discards any hand edits.
    ///
    /// Invoke: unity command eval --code "PoRacer.EditorTools.Editor_BakeAuthoredTrack.Bake()"
    ///         (or -executeMethod PoRacer.EditorTools.Editor_BakeAuthoredTrack.Bake)
    /// Once baked, the subtrees are ordinary GameObjects - edit them with the MCP scene
    /// commands (find_gameobjects / set_component_properties) rather than re-running this.
    /// </summary>
    public static class Editor_BakeAuthoredTrack
    {
        private const string SCENE_PATH = "Assets/Scenes/SCN_RACE_FLAT.unity";
        private const string AUTHORED_PREFIX = "AuthoredTrack_";

        // Must match what Systems_Spawn passes. Width and backMargin are constants
        // there; backMargin resolves to CAMERA_BACKDROP_MARGIN at every roster
        // size, because the deepest grid (3 rows x 1.6 m + 4) is 8.8 m and never
        // reaches 24.
        private const float TRACK_WIDTH = 24f;
        private const float BACK_MARGIN = 24f;
        private const int DECOR_SEED = 20260903;

        public static void Bake()
        {
            string error = BakeInternal();
            if (error != null)
            {
                Debug.LogError("Editor_BakeAuthoredTrack: " + error);
            }
        }

        private static string BakeInternal()
        {
            // EditorSceneManager.OpenScene throws in play mode, and the throw
            // reads as a Unity internal error rather than as "you are in play
            // mode". Bake writes a scene asset, so it belongs in edit mode only.
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                return "cannot bake while in play mode - exit play mode and run this again";
            }

            Scene scene = EditorSceneManager.OpenScene(SCENE_PATH, OpenSceneMode.Single);
            if (!scene.IsValid())
            {
                return "could not open " + SCENE_PATH;
            }

            RaceTrackView track = Object.FindFirstObjectByType<RaceTrackView>();
            if (track == null)
            {
                return "no RaceTrackView in " + SCENE_PATH;
            }
            if (track.TrackRoot == null)
            {
                return "RaceTrackView has no TrackRoot assigned";
            }

            // Replace every previous bake rather than layering a second one under it.
            for (int childIndex = track.transform.childCount - 1; childIndex >= 0; childIndex--)
            {
                Transform child = track.transform.GetChild(childIndex);
                if (child.name.StartsWith(AUTHORED_PREFIX))
                {
                    Object.DestroyImmediate(child.gameObject);
                }
            }

            var builder = new Systems_TrackBuilder(
                track.GroundMaterial, track.ObstacleMaterial, track.PhysicsMaterial);
            var baked = new List<RaceTrackView.AuthoredTrack>();
            var summary = new System.Text.StringBuilder();

            IReadOnlyList<Systems_MapCatalog.MapEntry> maps = Systems_MapCatalog.Entries;
            for (int mapIndex = 0; mapIndex < maps.Count; mapIndex++)
            {
                Systems_MapCatalog.MapEntry map = maps[mapIndex];
                if (map.Randomize || !map.Available)
                {
                    continue;
                }

                var authored = new GameObject(AUTHORED_PREFIX + map.DisplayName);
                authored.transform.SetParent(track.transform, worldPositionStays: false);
                authored.transform.localPosition = track.TrackRoot.localPosition;
                authored.transform.localRotation = track.TrackRoot.localRotation;
                authored.transform.localScale = track.TrackRoot.localScale;

                // The finish line sits at the map's length at race time; bake against
                // that position so the arch and decoration land where the race sees them.
                float finishZ = map.LengthMeters - 2f;

                // The parent is brand new, so the clear pass at the top of Build finds
                // no children - its Destroy would otherwise be illegal in edit mode.
                builder.Build(map.Kind, authored.transform, TRACK_WIDTH, map.LengthMeters,
                    new System.Random(DECOR_SEED + mapIndex), decorate: true, finishZ: finishZ,
                    features: map.Features, backMargin: BACK_MARGIN);

                if (!builder.TryGetGroundBounds(out Bounds groundBounds))
                {
                    Object.DestroyImmediate(authored);
                    return map.DisplayName + ": builder produced no ground bounds; nothing baked";
                }
                bool hasArch = builder.TryGetFinishArchBounds(out Bounds archBounds);

                int objectCount = authored.GetComponentsInChildren<Transform>(true).Length - 1;
                if (objectCount == 0)
                {
                    Object.DestroyImmediate(authored);
                    return map.DisplayName + ": builder produced no geometry; nothing baked";
                }

                // Only the first map is visible behind the menu; the race shows the rest.
                authored.SetActive(baked.Count == 0);
                baked.Add(new RaceTrackView.AuthoredTrack
                {
                    root = authored.transform,
                    kind = map.Kind,
                    lengthMeters = map.LengthMeters,
                    features = map.Features,
                    groundBounds = groundBounds,
                    hasArch = hasArch,
                    archBounds = archBounds,
                });
                summary.Append($"\n  {authored.name}: {objectCount} objects, ground {groundBounds.size}, " +
                    $"arch {(hasArch ? archBounds.size.ToString() : "none")}");
            }

            var serialized = new SerializedObject(track);
            SerializedProperty list = serialized.FindProperty("_authoredTracks");
            list.arraySize = baked.Count;
            for (int bakedIndex = 0; bakedIndex < baked.Count; bakedIndex++)
            {
                SerializedProperty element = list.GetArrayElementAtIndex(bakedIndex);
                RaceTrackView.AuthoredTrack entry = baked[bakedIndex];
                element.FindPropertyRelative("root").objectReferenceValue = entry.root;
                element.FindPropertyRelative("kind").enumValueIndex = (int)entry.kind;
                element.FindPropertyRelative("lengthMeters").floatValue = entry.lengthMeters;
                element.FindPropertyRelative("features").intValue = (int)entry.features;
                element.FindPropertyRelative("groundBounds").boundsValue = entry.groundBounds;
                element.FindPropertyRelative("hasArch").boolValue = entry.hasArch;
                element.FindPropertyRelative("archBounds").boundsValue = entry.archBounds;
            }
            serialized.ApplyModifiedPropertiesWithoutUndo();

            EditorSceneManager.MarkSceneDirty(scene);
            if (!EditorSceneManager.SaveScene(scene))
            {
                return "SaveScene failed for " + SCENE_PATH;
            }

            Debug.Log($"Baked {baked.Count} authored tracks into {SCENE_PATH}:{summary}\n" +
                "Systems_Spawn now skips the runtime builder for these maps; edit the subtrees by hand.");
            return null;
        }
    }
}
#endif
