using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// One MuJoCo &lt;position&gt; servo: force = clip(kp (ctrl - q) - kv qdot, forcerange),
    /// ctrl in radians. The rig lists these in action order: action i drives actuator i.
    /// </summary>
    public sealed class CreatureActuatorDef
    {
        public string Name { get; set; } = string.Empty;
        /// <summary>Index into the rig's joints.</summary>
        public int Joint { get; set; }
        public float Kp { get; set; }
        public float Kv { get; set; }
        public bool CtrlLimited { get; set; }
        public Vector2 CtrlRange { get; set; }
        public bool ForceLimited { get; set; }
        public Vector2 ForceRange { get; set; }
    }
}
