using System;
using System.IO;
using System.Linq;
using Creature;
using Mujoco;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace CreatureEditor {

/// <summary>
/// Builds a ready-to-play scene that verifies the trained policy in Unity.
///
/// Invoke: unity command eval --code "CreatureEditor.CreatureSceneBuilder.BuildFromMenu()"
/// (or Unity -batchmode -quit -executeMethod CreatureEditor.CreatureSceneBuilder.Build)
/// The [MenuItem] was removed on 2026-09-03 so all editor work goes through MCP / the CLI.
///
/// Everything the scene needs is set here on purpose -- especially Fixed
/// Timestep, which the plug-in takes from Time.fixedDeltaTime and NOT from the
/// MJCF. Leave it at Unity's 0.02 default and the policy runs at 10 Hz instead
/// of the 50 Hz it trained at, and the creature flails for no visible reason.
/// </summary>
public static class CreatureSceneBuilder {

  const string Xml = "Assets/Creature/creature.xml";
  const string PolicyJson = "Assets/Creature/policy.json";
  const string ScenePath = "Assets/Creature/CreatureVerification.unity";
  const float SimTimestep = 0.004f;   // must match <option timestep> in creature.xml

  // Ground grid. The texture spans 2 m so a 2x2 checker gives 1 m cells.
  const float GridTextureMetres = 2f;
  const int PixelsPerMetre = 128;
  const float GroundHalf = 40f;       // push the plane forward; the creature runs +Z

  public static void BuildFromMenu() {
    if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) {
      return;
    }
    string path = BuildInternal();
    if (path != null) {
      EditorUtility.DisplayDialog(
          "Creature verification scene",
          "Scene built and opened:\n\n" + path +
          "\n\nFixed Timestep set to " + SimTimestep + " s.\nPress Play to watch it run.",
          "OK");
    }
  }

  /// <summary>Entry point for -executeMethod. Exits with a status code.</summary>
  public static void Build() {
    int code = BuildInternal() != null ? 0 : 1;
    if (Application.isBatchMode) {
      EditorApplication.Exit(code);
    }
  }

  static string BuildInternal() {
    try {
      if (!File.Exists(Xml)) {
        Debug.LogError($"CreatureSceneBuilder: {Xml} not found."); return null;
      }
      var policy = AssetDatabase.LoadAssetAtPath<TextAsset>(PolicyJson);
      if (policy == null) {
        Debug.LogError(
            $"CreatureSceneBuilder: {PolicyJson} not found. Train first, then run " +
            "mjx_training/export_policy.py --run <name>.");
        return null;
      }

      // The plug-in reads the physics rate from here, not from the MJCF.
      // NOTE: this is a PROJECT setting, not a scene one -- it does not travel
      // with the saved scene, which is why CreatureSceneBootstrap also sets it
      // at load time. Both, so the Editor is right now and elsewhere later.
      Time.fixedDeltaTime = SimTimestep;

      var scene = EditorSceneManager.NewScene(
          NewSceneSetup.EmptyScene, NewSceneMode.Single);

      // --- the creature -------------------------------------------------
      var root = new MjImporterWithAssets().ImportFile(Path.GetFullPath(Xml));
      if (root == null) {
        Debug.LogError("CreatureSceneBuilder: MJCF import returned null."); return null;
      }

      // The importer may add its own MjScene; MjScene.Instance would add another.
      // Two of them makes the plug-in throw on load, so collapse to exactly one.
      var existing = UnityEngine.Object.FindObjectsByType<MjScene>(FindObjectsInactive.Include);
      Debug.Log($"CreatureSceneBuilder: MjScene count after import = {existing.Length}");
      for (int i = 1; i < existing.Length; i++) {
        UnityEngine.Object.DestroyImmediate(existing[i]);   // component only, keep the GameObject
      }
      var mjScene = existing.Length > 0 ? existing[0] : MjScene.Instance;

      var torso = UnityEngine.Object
          .FindObjectsByType<MjBody>(FindObjectsInactive.Include)
          .FirstOrDefault(b => StripId(b.gameObject.name) == "torso");
      if (torso == null) {
        Debug.LogError("CreatureSceneBuilder: no body named 'torso' after import."); return null;
      }

      // --- the agent ----------------------------------------------------
      var agent = root.AddComponent<CreatureAgent>();
      agent.policyJson = policy;
      agent.torso = torso;
      agent.applyHomePose = true;   // Unity's importer drops <keyframe>
      agent.logBindings = true;

      root.AddComponent<CreatureHud>();

      var boot = new GameObject("CreatureSceneBootstrap").AddComponent<CreatureSceneBootstrap>();
      boot.fixedTimestep = SimTimestep;

      // --- materials -----------------------------------------------------
      // The plug-in generates materials for the built-in pipeline. In a URP
      // project those shaders do not resolve and every geom renders magenta,
      // so build pipeline-appropriate materials and assign them explicitly.
      var bodyMat = MakeMaterial(new Color(0.92f, 0.55f, 0.20f), "CreatureBody");
      var legMat  = MakeMaterial(new Color(0.55f, 0.36f, 0.20f), "CreatureLeg");
      var groundMat = MakeMaterial(new Color(0.24f, 0.27f, 0.32f), "CreatureGround");

      int painted = 0;
      foreach (var r in root.GetComponentsInChildren<MeshRenderer>(true)) {
        string n = r.gameObject.name.ToLowerInvariant();
        r.sharedMaterial = (n.Contains("shin") || n.Contains("thigh")) ? legMat : bodyMat;
        painted++;
      }
      Debug.Log($"CreatureSceneBuilder: assigned materials to {painted} renderers " +
                $"(shader '{bodyMat.shader.name}')");

      // MuJoCo's infinite plane has no usable mesh, so add a visual ground.
      // Physics still belongs to MuJoCo -- this carries no collider.
      // A Unity Plane is 10 units across at scale 1, so scale S spans 10*S metres.
      const float groundScale = 12f;             // 120 m x 120 m
      const float groundMetres = 10f * groundScale;
      var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
      ground.name = "GroundVisual";
      ground.transform.localScale = new Vector3(groundScale, 1f, groundScale);
      ground.transform.position = new Vector3(0f, 0f, GroundHalf);

      // The texture covers 2 m x 2 m (a 2x2 checker of 1 m cells), so tiling it
      // groundMetres/2 times makes every visible square exactly one square metre.
      var gridTex = MakeGridTexture();
      float tiles = groundMetres / GridTextureMetres;
      ApplyTexture(groundMat, gridTex, new Vector2(tiles, tiles));
      ground.GetComponent<MeshRenderer>().sharedMaterial = groundMat;
      foreach (var c in ground.GetComponents<Collider>()) {
        UnityEngine.Object.DestroyImmediate(c);
      }

      // --- camera, light, sky -------------------------------------------
      var camGo = new GameObject("Main Camera", typeof(Camera), typeof(AudioListener));
      camGo.tag = "MainCamera";
      var cam = camGo.GetComponent<Camera>();
      cam.backgroundColor = new Color(0.16f, 0.19f, 0.24f);
      cam.clearFlags = CameraClearFlags.SolidColor;
      cam.farClipPlane = 200f;
      camGo.transform.position = new Vector3(-2.2f, 1.3f, 3.2f);
      var follow = camGo.AddComponent<CreatureCameraFollow>();
      follow.target = torso.transform;

      var lightGo = new GameObject("Directional Light", typeof(Light));
      var light = lightGo.GetComponent<Light>();
      light.type = LightType.Directional;
      light.intensity = 1.1f;
      light.shadows = LightShadows.Soft;
      lightGo.transform.rotation = Quaternion.Euler(48f, -35f, 0f);

      RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
      RenderSettings.ambientLight = new Color(0.35f, 0.38f, 0.44f);

      EditorSceneManager.MarkSceneDirty(scene);
      Directory.CreateDirectory(Path.GetDirectoryName(ScenePath));
      if (!EditorSceneManager.SaveScene(scene, ScenePath)) {
        Debug.LogError("CreatureSceneBuilder: SaveScene failed."); return null;
      }
      AssetDatabase.Refresh();

      var bodies = UnityEngine.Object
          .FindObjectsByType<MjBody>(FindObjectsInactive.Include).Length;
      var acts = UnityEngine.Object
          .FindObjectsByType<MjActuator>(FindObjectsInactive.Include).Length;
      Debug.Log($"CreatureSceneBuilder: built {ScenePath} | bodies={bodies} actuators={acts} " +
                $"fixedDeltaTime={Time.fixedDeltaTime} policy={policy.name} " +
                $"mjScenes={UnityEngine.Object.FindObjectsByType<MjScene>(FindObjectsInactive.Include).Length}");
      return ScenePath;
    } catch (Exception e) {
      Debug.LogError("CreatureSceneBuilder failed: " + e);
      return null;
    }
  }

  /// <summary>
  /// A 2 m x 2 m tile holding a 2x2 checker of 1 m cells with grid lines, so a
  /// tiled ground reads as exact square metres. Every tenth line is brighter,
  /// which makes 10 m intervals countable at a glance.
  /// </summary>
  static Texture2D MakeGridTexture() {
    int size = (int)(GridTextureMetres * PixelsPerMetre);   // 256
    int cell = PixelsPerMetre;                              // 128 px == 1 m
    var tex = new Texture2D(size, size, TextureFormat.RGBA32, true);
    tex.name = "CreatureGroundGrid";
    tex.wrapMode = TextureWrapMode.Repeat;
    tex.filterMode = FilterMode.Trilinear;
    tex.anisoLevel = 8;

    var light = new Color(0.30f, 0.34f, 0.40f);
    var dark = new Color(0.25f, 0.28f, 0.34f);
    var line = new Color(0.52f, 0.58f, 0.66f);
    int linePx = Mathf.Max(2, PixelsPerMetre / 48);         // ~2-3 px

    var px = new Color[size * size];
    for (int y = 0; y < size; y++) {
      for (int x = 0; x < size; x++) {
        bool checker = ((x / cell) + (y / cell)) % 2 == 0;
        var c = checker ? light : dark;
        // draw the 1 m boundaries
        int mx = x % cell, my = y % cell;
        if (mx < linePx || my < linePx || mx >= cell - linePx || my >= cell - linePx) {
          c = line;
        }
        px[y * size + x] = c;
      }
    }
    tex.SetPixels(px);
    tex.Apply(true);

    Directory.CreateDirectory("Assets/Creature/Materials");
    string path = AssetDatabase.GenerateUniqueAssetPath(
        "Assets/Creature/Materials/CreatureGroundGrid.asset");
    AssetDatabase.CreateAsset(tex, path);
    return tex;
  }

  /// <summary>Assigns a texture and tiling across built-in and URP/HDRP property names.</summary>
  static void ApplyTexture(Material mat, Texture tex, Vector2 tiling) {
    foreach (var prop in new[] { "_BaseMap", "_MainTex" }) {
      if (mat.HasProperty(prop)) {
        mat.SetTexture(prop, tex);
        mat.SetTextureScale(prop, tiling);
      }
    }
    // A textured surface should not also be tinted dark.
    if (mat.HasProperty("_BaseColor")) { mat.SetColor("_BaseColor", Color.white); }
    if (mat.HasProperty("_Color")) { mat.SetColor("_Color", Color.white); }
  }

  /// <summary>A material on whichever pipeline this project actually uses.</summary>
  static Material MakeMaterial(Color color, string name) {
    var shader = Shader.Find("Universal Render Pipeline/Lit")
                 ?? Shader.Find("HDRP/Lit")
                 ?? Shader.Find("Standard");
    var mat = new Material(shader) { name = name };
    if (mat.HasProperty("_BaseColor")) { mat.SetColor("_BaseColor", color); }
    if (mat.HasProperty("_Color")) { mat.SetColor("_Color", color); }
    if (mat.HasProperty("_Smoothness")) { mat.SetFloat("_Smoothness", 0.2f); }
    if (mat.HasProperty("_Glossiness")) { mat.SetFloat("_Glossiness", 0.2f); }

    Directory.CreateDirectory("Assets/Creature/Materials");
    string path = $"Assets/Creature/Materials/{name}.mat";
    AssetDatabase.CreateAsset(mat, AssetDatabase.GenerateUniqueAssetPath(path));
    return mat;
  }

  /// <summary>MjScene.CreateScene() appends "_&lt;id&gt;" to make names unique.</summary>
  static string StripId(string name) {
    int i = name.LastIndexOf('_');
    if (i <= 0 || i == name.Length - 1) {
      return name;
    }
    for (int k = i + 1; k < name.Length; k++) {
      if (!char.IsDigit(name[k])) {
        return name;
      }
    }
    return name.Substring(0, i);
  }
}

}  // namespace CreatureEditor
