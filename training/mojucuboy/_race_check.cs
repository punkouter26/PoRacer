var projectRoot = System.IO.Directory.GetParent(UnityEngine.Application.dataPath).FullName;
var outPath = System.IO.Path.Combine(projectRoot, "training", "mojucuboy", "_race_check.txt");
var sb = new System.Text.StringBuilder();
var inactive = UnityEngine.FindObjectsInactive.Include;
sb.Append("playing=").Append(UnityEngine.Application.isPlaying)
  .Append(" fixedDeltaTime=").Append(UnityEngine.Time.fixedDeltaTime).Append('\n');
sb.Append("MjScene instances=").Append(UnityEngine.Object.FindObjectsByType<Mujoco.MjScene>(inactive).Length)
  .Append(" (must be exactly 1)\n");
var agents = UnityEngine.Object.FindObjectsByType<PoRacer.Agents.Agent_MojucuBoy>(inactive);
sb.Append("Agent_MojucuBoy spawned=").Append(agents.Length).Append('\n');
foreach (var a in agents) {
    var body = a.Body;
    sb.Append("  ").Append(a.name)
      .Append(" failed=").Append(a.Failed)
      .Append(" bodyY=").Append(body.position.y.ToString("F3"))
      .Append(" bodyZ=").Append(body.position.z.ToString("F2"))
      .Append(" upright=").Append(UnityEngine.Vector3.Dot(body.up, UnityEngine.Vector3.up).ToString("F2"))
      .Append(" rootIsContainer=").Append(body != a.transform)
      .Append('\n');
    var smr = a.GetComponentInChildren<UnityEngine.SkinnedMeshRenderer>(true);
    if (smr != null) {
        var blk = new UnityEngine.MaterialPropertyBlock();
        smr.GetPropertyBlock(blk);
        sb.Append("    skin '").Append(smr.name).Append("' hasTintBlock=").Append(!blk.isEmpty)
          .Append(" (false = original texture kept)\n");
    }
}
var fidos = UnityEngine.Object.FindObjectsByType<PoRacer.Agents.Agent_Fido>(inactive);
sb.Append("Agent_Fido spawned=").Append(fidos.Length).Append('\n');
System.IO.File.WriteAllText(outPath, sb.ToString());
return "ok";
