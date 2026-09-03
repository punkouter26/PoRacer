var projectRoot = System.IO.Directory.GetParent(UnityEngine.Application.dataPath).FullName;
var outPath = System.IO.Path.Combine(projectRoot, "training", "mojucuboy", "_diag.txt");
var sb = new System.Text.StringBuilder();
var inactive = UnityEngine.FindObjectsInactive.Include;

var fido = UnityEngine.Object.FindAnyObjectByType<PoRacer.Agents.Agent_Fido>();
if (fido != null) sb.Append("Fido bodyZ=").Append(fido.Body.position.z.ToString("F2"))
                    .Append(" bodyY=").Append(fido.Body.position.y.ToString("F3")).Append('\n');

var ctrl = UnityEngine.Object.FindAnyObjectByType<Creature.MojucuBoy.MojucuBoyController>();
sb.Append("MojucuBoyController found=").Append(ctrl != null);
if (ctrl != null) {
    sb.Append(" enabled=").Append(ctrl.enabled).Append(" activeInHierarchy=").Append(ctrl.gameObject.activeInHierarchy);
    var t = ctrl.GetType();
    foreach (var fn in new string[]{"_bound","_failed","_stepCounter","_passive"}) {
        var f = t.GetField(fn, System.Reflection.BindingFlags.NonPublic|System.Reflection.BindingFlags.Instance);
        sb.Append(' ').Append(fn).Append('=').Append(f == null ? "?" : f.GetValue(ctrl));
    }
}
sb.Append('\n');

// Are his MjComponents actually in the compiled model? A body that never got an
// id was not part of CreateScene.
int bound = 0, unbound = 0;
foreach (var b in UnityEngine.Object.FindObjectsByType<Mujoco.MjBody>(inactive)) {
    bool mine = b.transform.root.name.Contains("MojucuBoy");
    if (!mine) continue;
    if (b.MujocoId >= 0) bound++; else unbound++;
}
sb.Append("MojucuBoy MjBody bound=").Append(bound).Append(" unbound=").Append(unbound).Append('\n');
int abound = 0, aunbound = 0;
foreach (var a in UnityEngine.Object.FindObjectsByType<Mujoco.MjActuator>(inactive)) {
    if (!a.transform.root.name.Contains("MojucuBoy")) continue;
    if (a.MujocoId >= 0) abound++; else aunbound++;
}
sb.Append("MojucuBoy MjActuator bound=").Append(abound).Append(" unbound=").Append(aunbound).Append('\n');
System.IO.File.WriteAllText(outPath, sb.ToString());
return "ok";
