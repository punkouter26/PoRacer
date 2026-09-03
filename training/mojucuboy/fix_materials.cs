// org.mujoco's importer assigns the built-in "Standard" shader to every geom it
// creates. This project renders with URP, which does not support that shader and
// draws it magenta. Remap every imported geom renderer onto URP/Lit.
//
// Also nudges MjMeshFilter into rebuilding: it generates the primitive mesh in
// Update(), which does not tick while the Editor is being driven over the CLI, so
// the renderers otherwise hold an empty unnamed mesh and draw nothing.
var projectRoot = System.IO.Directory.GetParent(UnityEngine.Application.dataPath).FullName;
var outPath = System.IO.Path.Combine(projectRoot, "training", "boy", "_probe_materials.txt");
var sb = new System.Text.StringBuilder();

var urpLit = UnityEngine.Shader.Find("Universal Render Pipeline/Lit");
sb.Append("URP/Lit found: ").Append(urpLit != null).Append('\n');
if (urpLit == null) { System.IO.File.WriteAllText(outPath, sb.ToString()); return "no URP shader"; }

var mat = new UnityEngine.Material(urpLit);
mat.name = "MojucuBoyGeomDebug";
mat.SetColor("_BaseColor", new UnityEngine.Color(0.70f, 0.70f, 0.75f, 1f));
UnityEditor.AssetDatabase.CreateAsset(mat, "Assets/Agents/MojucuBoy_v01/MojucuBoyGeomDebug.mat");

int swapped = 0, rebuilt = 0;
var inactive = UnityEngine.FindObjectsInactive.Include;
foreach (var geom in UnityEngine.Object.FindObjectsByType<Mujoco.MjGeom>(inactive)) {
    var rend = geom.GetComponent<UnityEngine.MeshRenderer>();
    if (rend != null) { rend.sharedMaterial = mat; swapped++; }

    // Build the primitive mesh directly rather than waiting for an Update tick.
    var filter = geom.GetComponent<UnityEngine.MeshFilter>();
    if (filter != null && geom.ShapeType != Mujoco.MjShapeComponent.ShapeTypes.Mesh) {
        var data = geom.BuildMesh();
        if (data != null) {
            var mesh = new UnityEngine.Mesh();
            mesh.name = geom.name + "_shape";
            mesh.vertices = data.Item1;
            mesh.triangles = data.Item2;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            filter.sharedMesh = mesh;
            rebuilt++;
        }
    }
}
sb.Append("materials swapped to URP/Lit: ").Append(swapped).Append('\n');
sb.Append("primitive meshes built: ").Append(rebuilt).Append('\n');

var check = UnityEngine.Object.FindObjectsByType<UnityEngine.Renderer>(inactive);
var total = new UnityEngine.Bounds();
bool first = true;
foreach (var r in check) {
    if (r.bounds.size == UnityEngine.Vector3.zero) { continue; }
    if (first) { total = r.bounds; first = false; } else { total.Encapsulate(r.bounds); }
}
sb.Append("combined renderer bounds: centre=").Append(total.center.ToString("F3"))
  .Append(" size=").Append(total.size.ToString("F3")).Append('\n');

UnityEditor.SceneManagement.EditorSceneManager.SaveScene(
    UnityEngine.SceneManagement.SceneManager.GetActiveScene());
System.IO.File.WriteAllText(outPath, sb.ToString());
return sb.ToString();
