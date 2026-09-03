using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace IsaacBox.EditorTools
{
    /// <summary>
    /// Builds the IsaacBox's URP materials from the textures extracted out of
    /// <c>IsaacBox_Character.glb</c> and puts them on the prefab's skinned renderers.
    ///
    /// The FBX carries geometry and skinning but NO textures at all - no embedded images and
    /// no texture references - so the materials Unity imports from it are untextured, which is
    /// why the creature renders flat. The GLB twin carries all ten images and six materials.
    /// The extracted PNGs live in <c>Textures/</c> beside this port's other data.
    ///
    /// Assignment is by renderer GameObject name, not by material index: the FBX material
    /// order is an import detail that a re-import can reshuffle, while the names come from the
    /// authored mesh and are stable. Sole_L and Sole_R deliberately share one material, which
    /// is how the GLB has it (Shoe_Rubber is the one material with no texture).
    ///
    /// <see cref="Apply"/> is called from IsaacBoxSetup.BuildPrefab before the prefab is saved,
    /// so rebuilding the prefab cannot silently revert the creature to the untextured FBX
    /// materials.
    /// </summary>
    internal static class IsaacBoxMaterials
    {
        private const string TEX_DIR = "Assets/unity_export/IsaacBox/Textures";
        private const string MAT_DIR = "Assets/unity_export/IsaacBox/Materials";

        // Mobile budget. The 26 MB of PNG in the repo is SOURCE, not app size - Unity
        // compresses on build, and at these caps the whole creature costs roughly 2.8 MB of
        // ASTC with mips. Caps are per-texture because the authored art is not uniform:
        // the head colour and the metallic/roughness map both ship at 4096.
        private const int MAX_SIZE_DEFAULT = 2048;
        // Gloss/metallic is low-frequency data - 4096 of it is indistinguishable from 512 on
        // a creature this size, and 512 costs ~0.04 MB against ~0.62 MB.
        private const int MAX_SIZE_METALLIC_ROUGHNESS = 512;

        private static readonly int BaseMap = Shader.PropertyToID("_BaseMap");
        private static readonly int BaseColor = Shader.PropertyToID("_BaseColor");
        private static readonly int BumpMap = Shader.PropertyToID("_BumpMap");
        private static readonly int MetallicGlossMap = Shader.PropertyToID("_MetallicGlossMap");
        private static readonly int Smoothness = Shader.PropertyToID("_Smoothness");
        private static readonly int Metallic = Shader.PropertyToID("_Metallic");

        /// <summary>One authored material: renderer names, textures and the glTF PBR factors.</summary>
        private readonly struct Spec
        {
            public readonly string Name;
            public readonly string[] Renderers;
            public readonly string BaseColorTex;
            public readonly string NormalTex;
            public readonly float Roughness;
            public readonly Color Tint;

            public Spec(string name, string[] renderers, string baseTex, string normalTex,
                        float roughness, Color tint)
            {
                Name = name;
                Renderers = renderers;
                BaseColorTex = baseTex;
                NormalTex = normalTex;
                Roughness = roughness;
                Tint = tint;
            }
        }

        // Straight out of the GLB's materials array; roughness factors are its own numbers.
        private static readonly Spec[] SPECS =
        {
            // The head's authored metallic/roughness MAP is deliberately not used - see the
            // channel-packing note below. Its measured content is metallic ~0.02 and roughness
            // 0.65-0.93 (mean 0.80), so two scalars reproduce it and an 11 MB 4K map does not
            // earn its place.
            new Spec("M_IsaacBox_Head",  new[] { "3DModel_Custom" }, "Head_BaseColor", null, 0.80f, Color.white),
            new Spec("M_IsaacBox_Body",  new[] { "Body" },  "Body_BaseColor",  "Body_Normal",  0.48f, Color.white),
            new Spec("M_IsaacBox_Pants", new[] { "Pants" }, "Pants_BaseColor", "Pants_Normal", 0.85f, Color.white),
            new Spec("M_IsaacBox_Shirt", new[] { "Shirt" }, "Shirt_BaseColor", "Shirt_Normal", 0.92f, Color.white),
            new Spec("M_IsaacBox_Shoes", new[] { "Shoes" }, "Shoes_BaseColor", "Shoes_Normal", 0.80f, Color.white),
            // Shoe_Rubber has no maps in the GLB, only a base colour factor.
            new Spec("M_IsaacBox_SoleRubber", new[] { "Sole_L", "Sole_R" }, null, null, 0.55f,
                     new Color(0.42f, 0.40f, 0.36f, 1f)),
        };

        [MenuItem("IsaacBox/Rebuild Materials From GLB Textures", priority = 4)]
        public static void RebuildMenu()
        {
            if (!Directory.Exists(TEX_DIR))
            {
                Debug.LogError($"[IsaacBox] {TEX_DIR} not found. Extract the GLB textures first.");
                return;
            }

            ConfigureTextureImports();
            Material[] mats = BuildMaterials();

            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(IsaacBoxPaths.Prefab);
            if (prefab == null)
            {
                Debug.LogWarning($"[IsaacBox] materials built, but {IsaacBoxPaths.Prefab} does not exist yet. " +
                                 "Run IsaacBox > Build Prefab.");
                return;
            }

            // Instantiating runs the agent's Awake in edit mode, and that Awake can disable the
            // component; saving the instance straight back would persist the disabled state into
            // the prefab. Guarantee it stays enabled before writing.
            GameObject instance = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
            int assigned = Apply(instance);
            var agent = instance.GetComponent<IsaacBoxAgent>();
            if (agent != null) agent.enabled = true;
            PrefabUtility.SaveAsPrefabAsset(instance, IsaacBoxPaths.Prefab);
            Object.DestroyImmediate(instance);
            AssetDatabase.SaveAssets();

            Debug.Log($"[IsaacBox] materials rebuilt: {mats.Length} materials in {MAT_DIR}, " +
                      $"{assigned} renderers assigned.");
        }

        /// <summary>
        /// Puts the authored materials on <paramref name="root"/>'s skinned renderers.
        /// Returns how many renderers were assigned. Missing materials are a warning, not a
        /// throw: a rig without its skin must still build.
        /// </summary>
        internal static int Apply(GameObject root)
        {
            if (root == null) return 0;

            var byRenderer = new Dictionary<string, Material>();
            foreach (Spec spec in SPECS)
            {
                Material m = AssetDatabase.LoadAssetAtPath<Material>($"{MAT_DIR}/{spec.Name}.mat");
                if (m == null) continue;
                foreach (string r in spec.Renderers) byRenderer[r] = m;
            }
            if (byRenderer.Count == 0)
            {
                Debug.LogWarning("[IsaacBox] no authored materials found; leaving the FBX's own " +
                                 "(untextured) materials in place. Run IsaacBox > Rebuild Materials.");
                return 0;
            }

            int assigned = 0;
            var missing = new List<string>();
            foreach (var smr in root.GetComponentsInChildren<SkinnedMeshRenderer>(true))
            {
                if (byRenderer.TryGetValue(smr.gameObject.name, out Material m))
                {
                    smr.sharedMaterial = m;
                    assigned++;
                }
                else
                {
                    missing.Add(smr.gameObject.name);
                }
            }
            if (missing.Count > 0)
                Debug.LogWarning($"[IsaacBox] no authored material for renderer(s): {string.Join(", ", missing)}. " +
                                 "They keep the FBX material. Add them to IsaacBoxMaterials.SPECS.");
            return assigned;
        }

        /// <summary>Normal maps must import as NormalMap and the metallic/roughness map as linear.</summary>
        private static void ConfigureTextureImports()
        {
            foreach (string path in Directory.GetFiles(TEX_DIR, "*.png"))
            {
                string assetPath = path.Replace('\\', '/');
                var ti = AssetImporter.GetAtPath(assetPath) as TextureImporter;
                if (ti == null) continue;

                bool isNormal = assetPath.Contains("_Normal");
                bool isMetallicRoughness = assetPath.Contains("Metallic") || assetPath.Contains("Roughness");
                // Metallic/roughness is data, not colour: sampling it through sRGB would skew
                // every gloss value on the head.
                bool isLinearData = isNormal || isMetallicRoughness;
                int cap = isMetallicRoughness ? MAX_SIZE_METALLIC_ROUGHNESS : MAX_SIZE_DEFAULT;

                TextureImporterType wanted = isNormal ? TextureImporterType.NormalMap : TextureImporterType.Default;
                bool dirty = ti.textureType != wanted
                             || ti.sRGBTexture != !isLinearData
                             || ti.maxTextureSize != cap;
                if (!dirty) continue;

                ti.textureType = wanted;
                if (!isNormal) ti.sRGBTexture = !isLinearData;
                ti.maxTextureSize = cap;
                ti.SaveAndReimport();
            }
        }

        private static Material[] BuildMaterials()
        {
            Directory.CreateDirectory(MAT_DIR);
            Shader lit = Shader.Find("Universal Render Pipeline/Lit");
            if (lit == null)
            {
                Debug.LogError("[IsaacBox] URP/Lit shader not found.");
                return new Material[0];
            }

            var made = new List<Material>();
            foreach (Spec spec in SPECS)
            {
                string path = $"{MAT_DIR}/{spec.Name}.mat";
                Material m = AssetDatabase.LoadAssetAtPath<Material>(path);
                bool isNew = m == null;
                if (isNew) m = new Material(lit);
                m.shader = lit;

                m.SetColor(BaseColor, spec.Tint);
                m.SetFloat(Smoothness, 1f - spec.Roughness);   // glTF roughness -> URP smoothness
                m.SetFloat(Metallic, 0f);                       // every GLB material is metallicFactor 0

                SetTex(m, BaseMap, spec.BaseColorTex);
                if (SetTex(m, BumpMap, spec.NormalTex)) m.EnableKeyword("_NORMALMAP");
                else m.DisableKeyword("_NORMALMAP");

                // NEVER put a glTF metallicRoughness map into URP's _MetallicGlossMap: the
                // channel packing is different and the result is not subtly wrong, it is wildly
                // wrong. glTF stores roughness in G and metallic in B, leaving R unused; URP
                // reads metallic from R and smoothness from A. The head's map has R = 255 and
                // A = 255 throughout, so URP rendered it as metallic 1.0 / smoothness 1.0 - a
                // chrome mirror, which read on screen as a dark red shiny head. Roughness and
                // metallic come from the scalars above instead.
                m.SetTexture(MetallicGlossMap, null);
                m.DisableKeyword("_METALLICSPECGLOSSMAP");

                if (isNew) AssetDatabase.CreateAsset(m, path);
                else EditorUtility.SetDirty(m);
                made.Add(m);
            }
            AssetDatabase.SaveAssets();
            return made.ToArray();
        }

        private static bool SetTex(Material m, int prop, string textureName)
        {
            if (string.IsNullOrEmpty(textureName)) { m.SetTexture(prop, null); return false; }
            var t = AssetDatabase.LoadAssetAtPath<Texture2D>($"{TEX_DIR}/{textureName}.png");
            if (t == null)
            {
                Debug.LogWarning($"[IsaacBox] texture {textureName}.png missing in {TEX_DIR}.");
                m.SetTexture(prop, null);
                return false;
            }
            m.SetTexture(prop, t);
            return true;
        }
    }
}
