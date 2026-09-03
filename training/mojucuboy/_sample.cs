var projectRoot = System.IO.Directory.GetParent(UnityEngine.Application.dataPath).FullName;
var outPath = System.IO.Path.Combine(projectRoot, "training", "mojucuboy", "_sample.txt");
var a = UnityEngine.Object.FindAnyObjectByType<PoRacer.Agents.Agent_MojucuBoy>();
string line;
if (a == null) { line = "no MojucuBoy (menu?)"; }
else {
    var b = a.Body;
    var ctrl = a.GetComponent<Creature.MojucuBoy.MojucuBoyController>();
    var t = ctrl.GetType();
    var hf = t.GetField("_commandHeading", System.Reflection.BindingFlags.NonPublic|System.Reflection.BindingFlags.Instance);
    var sf = t.GetField("_stepCounter", System.Reflection.BindingFlags.NonPublic|System.Reflection.BindingFlags.Instance);
    float fwd = UnityEngine.Mathf.Atan2(b.forward.z, b.forward.x);
    float cmd = hf == null ? 0f : (float)hf.GetValue(ctrl);
    float err = UnityEngine.Mathf.DeltaAngle(fwd * UnityEngine.Mathf.Rad2Deg, cmd * UnityEngine.Mathf.Rad2Deg);
    line = string.Format("t={0,6:F1}s pos=({1,6:F2},{2,5:F2},{3,6:F2}) upright={4,5:F2} failed={5,-5} cmdHdgErr={6,7:F1}deg steps={7}",
        UnityEngine.Time.timeSinceLevelLoad, b.position.x, b.position.y, b.position.z,
        UnityEngine.Vector3.Dot(b.up, UnityEngine.Vector3.up), a.Failed, err, sf == null ? -1 : sf.GetValue(ctrl));
}
System.IO.File.AppendAllText(outPath, line + "\n");
return line;
