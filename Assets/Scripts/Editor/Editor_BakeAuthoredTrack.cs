#if UNITY_EDITOR
using PoRacer.Systems;
using PoRacer.Views;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace PoRacer.EditorTools
{
    /// <summary>
    /// Bakes the Flat map into SCN_RACE_FLAT as ordinary scene objects, once,
    /// so it can be tuned by hand instead of being regenerated on every race.
    ///
    /// Systems_TrackBuilder still owns the other maps — Lumpy and Swamp have
    /// terrain meshes and scattered hazards, and Roulette re-rolls both every
    /// race, so those stay procedural. Flat is the one map with nothing random
    /// in it, which is what makes it worth freezing: the geometry it produced
    /// was identical every time, and nobody could move a crowd stand.
    ///
    /// Determinism. The generated version varied only in its decoration, which
    /// is drawn from the per-race RNG. This bakes one roll from a fixed seed and
    /// keeps it; after that the scene is the source of truth and re-running this
    /// discards any hand edits, which is why it asks first.
    ///
    /// Headless: unity cmd menu --path "PoRacer/Track/Bake Authored Flat Track"
    ///           (or -executeMethod PoRacer.EditorTools.Editor_BakeAuthoredTrack.Bake)
    /// </summary>
    public static class Editor_BakeAuthoredTrack
    {
        private const string SCENE_PATH = "Assets/Scenes/SCN_RACE_FLAT.unity";
        private const string AUTHORED_NAME = "AuthoredTrack_Flat";

        // Must match what Systems_Spawn passes for the Flat map. Width and
        // backMargin are constants there; backMargin resolves to
        // CAMERA_BACKDROP_MARGIN at every roster size, because the deepest grid
        // (3 rows x 1.6 m + 4) is 8.8 m and never reaches 24.
        private const float TRACK_WIDTH = 24f;
        private const float TRACK_LENGTH = 22f;
        private const float BACK_MARGIN = 24f;
        private const TrackKind KIND = TrackKind.Flat;
        private const TrackFeatures FEATURES = TrackFeatures.None;
        private const int DECOR_SEED = 20260903;

        [MenuItem("PoRacer/Track/Bake Authored Flat Track")]
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

            // Replace any previous bake rather than layering a second one under it.
            Transform existing = track.transform.Find(AUTHORED_NAME);
            if (existing != null)
            {
                Object.DestroyImmediate(existing.gameObject);
            }

            var authored = new GameObject(AUTHORED_NAME);
            authored.transform.SetParent(track.transform, worldPositionStays: false);
            authored.transform.localPosition = track.TrackRoot.localPosition;
            authored.transform.localRotation = track.TrackRoot.localRotation;
            authored.transform.localScale = track.TrackRoot.localScale;

            var builder = new Systems_TrackBuilder(
                track.GroundMaterial, track.ObstacleMaterial, track.PhysicsMaterial);
            float finishZ = track.FinishLine != null ? track.FinishLine.position.z : -1f;

            // The parent is brand new, so the clear pass at the top of Build finds
            // no children — its Destroy would otherwise be illegal in edit mode.
            builder.Build(KIND, authored.transform, TRACK_WIDTH, TRACK_LENGTH,
                new System.Random(DECOR_SEED), decorate: true, finishZ: finishZ,
                features: FEATURES, backMargin: BACK_MARGIN);

            if (!builder.TryGetGroundBounds(out Bounds groundBounds))
            {
                Object.DestroyImmediate(authored);
                return "builder produced no ground bounds; nothing baked";
            }
            bool hasArch = builder.TryGetFinishArchBounds(out Bounds archBounds);

            int objectCount = authored.GetComponentsInChildren<Transform>(true).Length - 1;
            if (objectCount == 0)
            {
                Object.DestroyImmediate(authored);
                return "builder produced no geometry; nothing baked";
            }

            var serialized = new SerializedObject(track);
            serialized.FindProperty("_authoredTrack").objectReferenceValue = authored.transform;
            serialized.FindProperty("_authoredKind").enumValueIndex = (int)KIND;
            serialized.FindProperty("_authoredLengthMeters").floatValue = TRACK_LENGTH;
            serialized.FindProperty("_authoredFeatures").intValue = (int)FEATURES;
            serialized.FindProperty("_authoredGroundBounds").boundsValue = groundBounds;
            serialized.FindProperty("_authoredHasArch").boolValue = hasArch;
            serialized.FindProperty("_authoredArchBounds").boundsValue = archBounds;
            serialized.ApplyModifiedPropertiesWithoutUndo();

            EditorSceneManager.MarkSceneDirty(scene);
            if (!EditorSceneManager.SaveScene(scene))
            {
                return "SaveScene failed for " + SCENE_PATH;
            }

            Debug.Log($"Baked authored Flat track into {SCENE_PATH}: {objectCount} objects under " +
                $"{AUTHORED_NAME}, ground {groundBounds.size}, arch {(hasArch ? archBounds.size.ToString() : "none")}. " +
                "Systems_Spawn now skips the runtime builder for the Flat map; edit the subtree by hand.");
            return null;
        }
    }
}
#endif
