// Point the main camera at MojucuBoy so the capture actually shows him.
var a = UnityEngine.Object.FindAnyObjectByType<PoRacer.Agents.Agent_MojucuBoy>();
if (a == null) { return "no MojucuBoy in scene"; }
var cam = UnityEngine.Camera.main;
if (cam == null) { return "no main camera"; }
var b = a.Body;
cam.transform.position = b.position + new UnityEngine.Vector3(1.8f, 1.1f, -1.9f);
cam.transform.LookAt(b.position + UnityEngine.Vector3.up * 0.2f);
return "framed MojucuBoy at " + b.position.ToString("F2") + " failed=" + a.Failed;
