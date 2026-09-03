// Phase 2 round-trip: import the rig into a Unity scene through org.mujoco, then
// export the scene back out via MjScene.CreateScene and write it to disk.
//
// Run with:  unity cmd eval_file --file training/mojucuboy/unity_roundtrip.cs --timeout 300
//
// Uses ImportString, NOT ImportFile, and imports mojucuboy_unity.xml rather than
// mojucuboy.xml. ImportFile compiles the file and re-saves it through MuJoCo's own
// writer before importing, and that writer drops the `axis` attribute from every
// hinge whose axis is MuJoCo's default (0,0,1) -- which org.mujoco's importer
// then misreads as Unity +X. See make_unity_mjcf.py for the full characterisation.
//
// The eval host treats warnings as errors and compiles without /unsafe, so no
// obsolete overloads and no MjModel pointer access. Output goes to a file as well
// as the return value: the bridge reports a 5000 ms main-thread timeout on long
// evals even when they succeed, and the return value is lost when it does.

var projectRoot = System.IO.Directory.GetParent(UnityEngine.Application.dataPath).FullName;
var sourcePath = System.IO.Path.Combine(projectRoot, "Assets", "Agents", "MojucuBoy_v01", "mojucuboy_unity.xml");
var exportPath = System.IO.Path.Combine(projectRoot, "training", "boy", "mojucuboy_roundtrip.xml");
var logPath = System.IO.Path.Combine(projectRoot, "training", "boy", "_roundtrip_log.txt");
var scenePath = "Assets/Agents/MojucuBoy_v01/SCN_MOJUCUBOY_RIGTEST.unity";
var log = new System.Text.StringBuilder();

// A fresh, empty scene: MjScene is a singleton, and any stray MjComponent left in
// an open scene would be swept into CreateScene's export.
var scene = UnityEditor.SceneManagement.EditorSceneManager.NewScene(
    UnityEditor.SceneManagement.NewSceneSetup.EmptyScene,
    UnityEditor.SceneManagement.NewSceneMode.Single);

// MjImporterWithAssets writes generated materials to Assets/Local/MjImports/<name>
// and THROWS if one already exists, so a second run of this script would fail. It
// then swallows that exception inside ImportString and returns null. Clear the
// directory first to make the round trip repeatable.
var importAssets = "Assets/Local/MjImports/mojucuboy";
if (UnityEditor.AssetDatabase.IsValidFolder(importAssets)) {
    UnityEditor.AssetDatabase.DeleteAsset(importAssets);
    UnityEditor.AssetDatabase.Refresh();
    log.Append("cleared ").Append(importAssets).Append('\n');
}

// The native plug-in binding does not survive a domain reload on its own, and
// ImportString -- unlike ImportFile -- never loads it.
Mujoco.MjEngineTool.LoadPlugins();

var importer = new Mujoco.MjImporterWithAssets();
var mjcf = System.IO.File.ReadAllText(sourcePath);
var root = importer.ImportString(mjcf, "mojucuboy", sourcePath);
log.Append("imported root: ").Append(root == null ? "NULL" : root.name).Append('\n');

var inactive = UnityEngine.FindObjectsInactive.Include;
var bodies = UnityEngine.Object.FindObjectsByType<Mujoco.MjBody>(inactive);
var joints = UnityEngine.Object.FindObjectsByType<Mujoco.MjBaseJoint>(inactive);
var geoms = UnityEngine.Object.FindObjectsByType<Mujoco.MjGeom>(inactive);
var acts = UnityEngine.Object.FindObjectsByType<Mujoco.MjActuator>(inactive);
log.Append("components: bodies=").Append(bodies.Length)
   .Append(" joints=").Append(joints.Length)
   .Append(" geoms=").Append(geoms.Length)
   .Append(" actuators=").Append(acts.Length).Append('\n');

// Bail out BEFORE saving the scene or writing the export. ImportString catches its
// own exceptions and returns null, so without this guard a failed import quietly
// saves an empty scene over the good one and exports an empty MJCF that then reads
// as a passing round trip.
if (root == null || bodies.Length == 0 || acts.Length != 21) {
    log.Append("ABORT: import failed -- scene and export left untouched.\n")
       .Append("Check the Unity console for the exception ImportString swallowed.\n");
    System.IO.File.WriteAllText(logPath, log.ToString());
    return log.ToString();
}

// Defect probe: did the position gains survive the importer?
foreach (var act in acts) {
    if (act.name.Contains("abdomen_z") || act.name.Contains("knee_L")) {
        log.Append("  actuator ").Append(act.name)
           .Append(" type=").Append(act.Type)
           .Append(" gainprm=[").Append(string.Join(",", act.CustomParams.GainPrm))
           .Append("] biasprm=[").Append(string.Join(",", act.CustomParams.BiasPrm))
           .Append("] forceRange=").Append(act.CommonParams.ForceRange)
           .Append('\n');
    }
}

// Defect probe: hinge axes, as Unity now holds them.
foreach (var joint in joints) {
    var hinge = joint as Mujoco.MjHingeJoint;
    if (hinge == null) { continue; }
    log.Append("  hinge ").Append(hinge.name.PadRight(16))
       .Append(" unityAxis=").Append((hinge.transform.rotation * UnityEngine.Vector3.right).ToString("F3"))
       .Append(" range=").Append(hinge.RangeLower.ToString("F2"))
       .Append("..").Append(hinge.RangeUpper.ToString("F2"))
       .Append('\n');
}

var settings = UnityEngine.Object.FindAnyObjectByType<Mujoco.MjGlobalSettings>();
log.Append("global settings present: ").Append(settings != null).Append('\n');
log.Append("Time.fixedDeltaTime: ").Append(UnityEngine.Time.fixedDeltaTime).Append('\n');

// Export with the authored names instead of org.mujoco's "<name>_<n>" scheme. The
// actuator order in mojucuboy_rig.json is a contract resolved BY NAME on both sides --
// the Python trainer and the Unity controller both call mj_name2id -- so a renamed
// export would force every lookup through a suffix-stripping guess.
if (settings != null) {
    settings.UseRawGameObjectNames = true;
    UnityEditor.EditorUtility.SetDirty(settings);
    log.Append("UseRawGameObjectNames = true\n");
}

// MjScene.Instance CREATES the singleton if absent, so read it once, deliberately,
// after the hierarchy is in place.
var mjScene = Mujoco.MjScene.Instance;
log.Append("MjScene: ").Append(mjScene.name).Append('\n');

// skipCompile:false so MuJoCo actually compiles what Unity generated -- a scene
// that exports but will not compile is not a passing round trip. CreateScene
// throws on a compile failure, so reaching the next line is the proof.
var doc = mjScene.CreateScene(false);
log.Append("CreateScene compiled OK\n");

var writerSettings = new System.Xml.XmlWriterSettings();
writerSettings.Indent = true;
writerSettings.IndentChars = "  ";
var writer = System.Xml.XmlWriter.Create(exportPath, writerSettings);
doc.Save(writer);
writer.Close();
log.Append("exported: ").Append(exportPath).Append('\n');

UnityEditor.SceneManagement.EditorSceneManager.SaveScene(scene, scenePath);
log.Append("scene saved: ").Append(scenePath).Append('\n');

System.IO.File.WriteAllText(logPath, log.ToString());
return log.ToString();
