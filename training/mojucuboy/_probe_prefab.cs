var sb = new System.Text.StringBuilder();
var p = UnityEditor.AssetDatabase.LoadAssetAtPath<UnityEngine.GameObject>("Assets/Prefabs/MojucuBoy_v01.prefab");
if (p == null) { return "prefab not found"; }
sb.Append("prefab: ").Append(p.name).Append('\n');
sb.Append("  MjBody      ").Append(p.GetComponentsInChildren<Mujoco.MjBody>(true).Length).Append('\n');
sb.Append("  MjHingeJoint").Append(p.GetComponentsInChildren<Mujoco.MjHingeJoint>(true).Length).Append('\n');
sb.Append("  MjFreeJoint ").Append(p.GetComponentsInChildren<Mujoco.MjFreeJoint>(true).Length).Append('\n');
sb.Append("  MjActuator  ").Append(p.GetComponentsInChildren<Mujoco.MjActuator>(true).Length).Append('\n');
sb.Append("  SkinnedMesh ").Append(p.GetComponentsInChildren<UnityEngine.SkinnedMeshRenderer>(true).Length).Append('\n');

int unbound = 0, wrong = 0;
foreach (var a in p.GetComponentsInChildren<Mujoco.MjActuator>(true)) {
    if (a.Joint == null) { unbound++; sb.Append("  UNBOUND actuator ").Append(a.name).Append('\n'); continue; }
    var expect = a.name.StartsWith("act_") ? a.name.Substring(4) : a.name;
    if (a.Joint.name != expect) { wrong++; sb.Append("  MISBOUND ").Append(a.name).Append(" -> ").Append(a.Joint.name).Append('\n'); }
}
sb.Append("  actuators unbound=").Append(unbound).Append(" misbound=").Append(wrong).Append('\n');

var ctrl = p.GetComponent<Creature.MojucuBoy.MojucuBoyController>();
var binder = p.GetComponent<Creature.MojucuBoy.MojucuBoyVisualBinder>();
var agent = p.GetComponent<PoRacer.Agents.Agent_MojucuBoy>();
sb.Append("  MojucuBoyController ").Append(ctrl != null).Append(" | VisualBinder ").Append(binder != null)
  .Append(" | Agent_MojucuBoy ").Append(agent != null).Append('\n');
if (ctrl != null) {
    var t = ctrl.GetType();
    var mf = t.GetField("_modelAsset", System.Reflection.BindingFlags.NonPublic|System.Reflection.BindingFlags.Instance);
    var rf = t.GetField("_rigJson", System.Reflection.BindingFlags.NonPublic|System.Reflection.BindingFlags.Instance);
    sb.Append("  modelAsset=").Append(mf != null && mf.GetValue(ctrl) != null ? ((UnityEngine.Object)mf.GetValue(ctrl)).name : "NULL")
      .Append(" | rigJson=").Append(rf != null && rf.GetValue(ctrl) != null ? ((UnityEngine.Object)rf.GetValue(ctrl)).name : "NULL").Append('\n');
}
sb.Append("  root renderers enabled: ");
int on = 0; foreach (var r in p.GetComponentsInChildren<UnityEngine.MeshRenderer>(true)) { if (r.enabled) on++; }
sb.Append(on).Append(" (collision geoms should be 0)\n");
return sb.ToString();
