#if UNITY_EDITOR
using System.Collections.Generic;
using PoRacer.Models;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace PoRacer.EditorTools
{
    /// <summary>
    /// Lists every model in the KIRI fruit-and-veg pack in Assets/Settings/FruitCatalog.asset
    /// and wires the asset into SCN_RACE_FLAT's GameLifetimeScope, so Systems_FruitPour
    /// has something to rain. Re-runnable: rewrites the list from what is on disk.
    ///
    /// The pack ships albedo and roughness only; its normal, AO and height maps were
    /// dropped on 2026-09-04 and every texture is capped at 512 in its importer meta,
    /// which is what keeps 141 scanned models at ~30 MB in the build.
    ///
    /// Invoke: unity command eval --code "return PoRacer.EditorTools.Editor_BuildFruitCatalog.Build();"
    /// </summary>
    public static class Editor_BuildFruitCatalog
    {
        private const string MODELS_FOLDER = "Assets/KIRI_Asset_Pack_Fruit_and_Veg/KIRI_Asset_Pack_Fruit_and_Veg_Low/Models";
        private const string CATALOG_PATH = "Assets/Settings/FruitCatalog.asset";
        private const string SCENE_PATH = "Assets/Scenes/SCN_RACE_FLAT.unity";

        public static string Build()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                return "cannot build while in play mode";
            }
            string[] guids = AssetDatabase.FindAssets("t:Model", new[] { MODELS_FOLDER });
            var models = new List<GameObject>(guids.Length);
            for (int guidIndex = 0; guidIndex < guids.Length; guidIndex++)
            {
                string path = AssetDatabase.GUIDToAssetPath(guids[guidIndex]);
                var model = AssetDatabase.LoadAssetAtPath<GameObject>(path);
                if (model != null && model.GetComponentInChildren<MeshFilter>() != null)
                {
                    models.Add(model);
                }
            }
            if (models.Count == 0)
            {
                return "no models under " + MODELS_FOLDER + " - is the pack imported?";
            }
            models.Sort((first, second) => string.CompareOrdinal(first.name, second.name));

            int madeReadable = EnsureReadable(models);

            var catalog = AssetDatabase.LoadAssetAtPath<FruitCatalog>(CATALOG_PATH);
            if (catalog == null)
            {
                catalog = ScriptableObject.CreateInstance<FruitCatalog>();
                AssetDatabase.CreateAsset(catalog, CATALOG_PATH);
            }
            var serialized = new SerializedObject(catalog);
            SerializedProperty list = serialized.FindProperty("_models");
            list.arraySize = models.Count;
            for (int modelIndex = 0; modelIndex < models.Count; modelIndex++)
            {
                list.GetArrayElementAtIndex(modelIndex).objectReferenceValue = models[modelIndex];
            }
            serialized.ApplyModifiedPropertiesWithoutUndo();
            AssetDatabase.SaveAssets();

            Scene scene = EditorSceneManager.OpenScene(SCENE_PATH, OpenSceneMode.Single);
            GameLifetimeScope scope = Object.FindFirstObjectByType<GameLifetimeScope>();
            if (scope == null)
            {
                return "catalog written with " + models.Count + " models, but no GameLifetimeScope in " + SCENE_PATH;
            }
            var scopeSerialized = new SerializedObject(scope);
            scopeSerialized.FindProperty("_fruitCatalog").objectReferenceValue = catalog;
            scopeSerialized.ApplyModifiedPropertiesWithoutUndo();
            EditorSceneManager.MarkSceneDirty(scene);
            if (!EditorSceneManager.SaveScene(scene))
            {
                return "SaveScene failed for " + SCENE_PATH;
            }
            return $"{CATALOG_PATH}: {models.Count} models, wired into {SCENE_PATH}" +
                   (madeReadable > 0 ? $", {madeReadable} made read/write" : ", all already read/write");
        }

        /// <summary>
        /// Turns Read/Write on for every model in the catalog, and returns how many
        /// needed it.
        ///
        /// WHY THIS IS NOT OPTIONAL. Systems_FruitPour builds each piece's collider at
        /// RUNTIME - it adds a MeshCollider and assigns filter.sharedMesh. Cooking that
        /// collider needs the mesh's vertex data on the CPU, and a player build throws
        /// that copy away after uploading to the GPU unless the mesh is marked
        /// readable. So on device the assignment silently produced a MeshCollider with
        /// no cooked geometry: the fruit had NO COLLISION AT ALL and fell straight
        /// through the road and the scene.
        ///
        /// It does not reproduce in the Editor, which keeps mesh data loaded for every
        /// asset regardless of the flag - so the shower looks correct right up until it
        /// is on a phone. That is the whole signature of this bug.
        ///
        /// The memory it costs is negligible here: the pack's meshes run 264 to 1804
        /// triangles (the "_low" scans), so the CPU copies are a rounding error against
        /// the textures. Anything authored to be collided with at runtime has to carry
        /// this flag; prefer this over rebuilding colliders as primitives, which would
        /// throw away the per-scan shape that makes a carrot skid and a pumpkin roll.
        /// </summary>
        private static int EnsureReadable(List<GameObject> models)
        {
            int changed = 0;
            for (int modelIndex = 0; modelIndex < models.Count; modelIndex++)
            {
                string path = AssetDatabase.GetAssetPath(models[modelIndex]);
                if (string.IsNullOrEmpty(path))
                {
                    continue;
                }
                if (AssetImporter.GetAtPath(path) is not ModelImporter importer || importer.isReadable)
                {
                    continue;
                }
                importer.isReadable = true;
                importer.SaveAndReimport();
                changed++;
            }
            return changed;
        }
    }
}
#endif
