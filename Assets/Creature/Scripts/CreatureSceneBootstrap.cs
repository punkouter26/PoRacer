using UnityEngine;

namespace Creature {

/// <summary>
/// Forces the physics rate this scene needs, at load time.
///
/// The MuJoCo plug-in ignores &lt;option timestep&gt; in the MJCF and steps once per
/// FixedUpdate, so the real physics rate is Time.fixedDeltaTime -- a PROJECT
/// setting, not a scene one. A scene alone therefore cannot guarantee its own
/// timestep: open it in a project set to Unity's 0.02 default and a policy
/// trained at 0.004 runs 5x too slow and just flails.
///
/// Setting it here means the scene is self-contained and correct wherever it is
/// opened. Execution order is forced early so this lands before MjScene.Awake.
/// </summary>
[DefaultExecutionOrder(-10000)]
public class CreatureSceneBootstrap : MonoBehaviour {

  [Tooltip("Physics timestep, in seconds. Must match <option timestep> in creature.xml.")]
  public float fixedTimestep = 0.004f;

  [Tooltip("Restore the project's previous value when the scene unloads.")]
  public bool restoreOnDestroy = true;

  [Tooltip("Log the change once, so a surprising physics rate is traceable.")]
  public bool logChange = true;

  float _previous = -1f;

  void Awake() {
    if (fixedTimestep <= 0f) {
      return;
    }
    _previous = Time.fixedDeltaTime;
    if (!Mathf.Approximately(_previous, fixedTimestep)) {
      Time.fixedDeltaTime = fixedTimestep;
      if (logChange) {
        Debug.Log(
            $"CreatureSceneBootstrap: Time.fixedDeltaTime {_previous:F4} -> {fixedTimestep:F4} s " +
            "(the MuJoCo plug-in takes its physics rate from here, not from the MJCF).",
            this);
      }
    }
  }

  void OnDestroy() {
    if (restoreOnDestroy && _previous > 0f) {
      Time.fixedDeltaTime = _previous;
    }
  }
}

}  // namespace Creature
