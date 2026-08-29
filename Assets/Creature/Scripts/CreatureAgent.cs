using System;
using System.Collections.Generic;
using System.Linq;
using Mujoco;
using UnityEngine;

namespace Creature {

/// <summary>
/// Drives the MuJoCo creature in Unity with a policy trained in MJX.
///
/// Observations are read straight out of mjData, so they are in MuJoCo's frame
/// and units -- identical to what the policy saw during training. The layout
/// must stay in lockstep with OBS_LAYOUT in mjx_training/creature_env.py:
///
///   [0..3)   gravity_local   world -Z rotated into the torso frame
///   [3..6)   linvel_local    qvel[0:3] (world) rotated into the torso frame
///   [6..9)   angvel_local    qvel[3:6], already torso-local for a free joint
///   [9..17)  joint_pos       qpos at each hinge, in actuator order
///   [17..25) joint_vel       qvel at each hinge, in actuator order
///   [25..33) last_action     previous action
/// </summary>
// Run after MjScene.Awake (order 0) has set its singleton. MjScene.Instance
// throws if an MjScene exists in the scene but Awake has not yet claimed it.
[DefaultExecutionOrder(100)]
public class CreatureAgent : MonoBehaviour {

  [Header("Policy")]
  [Tooltip("Assets/Creature/policy.json, produced by mjx_training/export_policy.py")]
  public TextAsset policyJson;

  [Tooltip("Physics steps each action is held for. Must equal nSubsteps used in " +
           "training (ctrl_dt / sim_dt). 0 = take the value from policy.json.")]
  public int actionDecimation = 0;

  [Header("Bindings")]
  [Tooltip("The creature root MjBody, the one carrying the free joint.")]
  public MjBody torso;

  [Tooltip("Hinge joints in MJCF actuator order. Leave empty to auto-resolve by name.")]
  public List<MjHingeJoint> joints = new List<MjHingeJoint>();

  [Tooltip("Actuators in MJCF order. Leave empty to auto-resolve by name.")]
  public List<MjActuator> actuators = new List<MjActuator>();

  [Header("Start pose")]
  [Tooltip("Apply the MJCF 'home' keyframe joint angles when the scene starts. " +
           "Unity's MJCF importer ignores <keyframe>, so without this the creature " +
           "begins with straight legs -- a stance the policy never trained from.")]
  public bool applyHomePose = true;

  [Header("Debug")]
  public bool logBindings = true;

  [Tooltip("Zero the controls instead of running the policy.")]
  public bool passive = false;

  /// <summary>MJCF actuator order. Must match the actuator block in creature.xml.</summary>
  public static readonly string[] ActuatorOrder = {
    "fl_hip", "fl_knee", "fr_hip", "fr_knee",
    "bl_hip", "bl_knee", "br_hip", "br_knee",
  };

  CreaturePolicy _policy;
  float[] _obs;
  float[] _action;
  float[] _lastAction;
  int[] _qposAddr;
  int[] _dofAddr;
  int _torsoId = -1;
  int _stepCounter;
  bool _bound;
  bool _failed;

  void OnEnable() {
    if (policyJson == null) {
      Debug.LogError(
          "CreatureAgent: no policyJson assigned. Train with mjx_training/train.py, " +
          "then run export_policy.py.", this);
      enabled = false;
      return;
    }
    try {
      _policy = CreaturePolicy.FromJson(policyJson.text);
    } catch (Exception e) {
      Debug.LogError("CreatureAgent: " + e.Message, this);
      enabled = false;
      return;
    }

    _obs = new float[_policy.ObsSize];
    _action = new float[_policy.ActionSize];
    _lastAction = new float[_policy.ActionSize];

    if (actionDecimation <= 0) {
      actionDecimation = Mathf.Max(1, _policy.NSubsteps);
    }

    var scene = ResolveScene();
    if (scene == null) {
      Debug.LogError("CreatureAgent: no MjScene in the scene. Import the MJCF via " +
                     "Assets > Import MuJoCo Scene first.", this);
      enabled = false;
      return;
    }
    scene.ctrlCallback += OnControl;
  }

  /// <summary>
  /// MjScene.Instance throws when a scene-authored MjScene exists but its Awake
  /// has not run yet (always the case in Edit mode, and possible in Play mode if
  /// this component initialises first). Fall back to finding it directly.
  /// </summary>
  static MjScene ResolveScene() {
    if (MjScene.InstanceExists) {
      return MjScene.Instance;
    }
    var found = FindObjectsByType<MjScene>(FindObjectsInactive.Include);
    if (found.Length > 0) {
      return found[0];
    }
    return MjScene.Instance;   // none in scene: let the plug-in create one
  }

  void OnDisable() {
    if (MjScene.InstanceExists) {
      MjScene.Instance.ctrlCallback -= OnControl;
    }
    _bound = false;
    _failed = false;
    _stepCounter = 0;
    if (_lastAction != null) {
      Array.Clear(_lastAction, 0, _lastAction.Length);
    }
  }

  // --- binding ---------------------------------------------------------
  static string NameOf(MjComponent c) {
    return string.IsNullOrEmpty(c.MujocoName) ? c.gameObject.name : c.MujocoName;
  }

  /// <summary>
  /// MjScene.CreateScene() uniquifies names by appending "_&lt;id&gt;" (so the MJCF
  /// actuator "fl_hip" becomes "fl_hip_30" once the scene is built). Strip a
  /// trailing all-digits segment so binding works both before and after that
  /// pass. A component genuinely named "foo_2" would resolve as "foo" -- keep
  /// digits out of the tail of MJCF names.
  /// </summary>
  static string BaseName(string name) {
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

  /// <summary>Indexes components under every name they might answer to.</summary>
  static Dictionary<string, T> IndexByName<T>(T[] items) where T : MjComponent {
    var map = new Dictionary<string, T>(StringComparer.Ordinal);
    // Exact names win; suffix-stripped names only fill gaps.
    foreach (var it in items) {
      map[NameOf(it)] = it;
      map[it.gameObject.name] = it;
    }
    foreach (var it in items) {
      var stripped = BaseName(NameOf(it));
      if (!map.ContainsKey(stripped)) {
        map[stripped] = it;
      }
      stripped = BaseName(it.gameObject.name);
      if (!map.ContainsKey(stripped)) {
        map[stripped] = it;
      }
    }
    return map;
  }

  bool TryBind() {
    if (torso == null) {
      torso = GetComponentInChildren<MjBody>();
    }
    if (torso == null) {
      Debug.LogError("CreatureAgent: no MjBody found. Assign the torso field.", this);
      return false;
    }
    _torsoId = torso.MujocoId;

    if (actuators.Count == 0) {
      var found = IndexByName(FindObjectsByType<MjActuator>(FindObjectsInactive.Include));
      foreach (var n in ActuatorOrder) {
        MjActuator a;
        if (!found.TryGetValue(n, out a)) {
          Debug.LogError(
              "CreatureAgent: no MjActuator named " + n + ". Found: " +
              string.Join(", ", found.Keys), this);
          return false;
        }
        actuators.Add(a);
      }
    }

    if (joints.Count == 0) {
      var found = IndexByName(FindObjectsByType<MjHingeJoint>(FindObjectsInactive.Include));
      // Hinge names match actuator names in creature.xml.
      foreach (var n in ActuatorOrder) {
        MjHingeJoint j;
        if (!found.TryGetValue(n, out j)) {
          Debug.LogError(
              "CreatureAgent: no MjHingeJoint named " + n + ". Found: " +
              string.Join(", ", found.Keys), this);
          return false;
        }
        joints.Add(j);
      }
    }

    if (actuators.Count != _policy.ActionSize || joints.Count != _policy.ActionSize) {
      Debug.LogError(
          "CreatureAgent: policy expects " + _policy.ActionSize +
          " actuators/joints, bound " + actuators.Count + "/" + joints.Count + ".", this);
      return false;
    }

    _qposAddr = new int[joints.Count];
    _dofAddr = new int[joints.Count];
    for (int i = 0; i < joints.Count; i++) {
      _qposAddr[i] = joints[i].QposAddress;
      _dofAddr[i] = joints[i].DofAddress;
      if (_qposAddr[i] < 0 || _dofAddr[i] < 0) {
        Debug.LogError(
            "CreatureAgent: joint " + NameOf(joints[i]) +
            " is not bound to the runtime yet.", this);
        return false;
      }
    }

    int expected = 9 + 2 * joints.Count + _policy.ActionSize;
    if (expected != _policy.ObsSize) {
      Debug.LogError(
          "CreatureAgent: observation layout mismatch -- this scene produces " +
          expected + " values but policy.json wants " + _policy.ObsSize + ".", this);
      return false;
    }

    if (logBindings) {
      Debug.Log(
          "CreatureAgent: bound torso " + NameOf(torso) + " (id " + _torsoId + "), " +
          actuators.Count + " actuators, decimation " + actionDecimation +
          ", obs " + _policy.ObsSize + ", act " + _policy.ActionSize + ".", this);
    }
    return true;
  }

  // --- control loop ----------------------------------------------------
  unsafe void OnControl(object sender, MjStepArgs e) {
    if (_failed) {
      return;
    }
    if (!_bound) {
      if (!TryBind()) {
        _failed = true;
        return;
      }
      _bound = true;
      CheckTiming(e.model);
      if (applyHomePose) {
        ApplyHomePose(e.model, e.data);
      }
    }

    if (passive) {
      for (int i = 0; i < actuators.Count; i++) {
        actuators[i].Control = 0f;
      }
      return;
    }

    if (_stepCounter % actionDecimation == 0) {
      BuildObservation(e.data);
      _policy.Evaluate(_obs, _action);
      Array.Copy(_action, _lastAction, _action.Length);
    }
    _stepCounter++;

    for (int i = 0; i < actuators.Count; i++) {
      actuators[i].Control = _lastAction[i];
    }
  }

  /// <summary>
  /// The MuJoCo Unity plug-in deliberately ignores the MJCF's &lt;option
  /// timestep&gt; and uses Time.fixedDeltaTime instead (MjGlobalSettings.cs). If
  /// that does not match what the policy trained with, the creature is being
  /// controlled at the wrong rate and will move badly for no visible reason.
  /// Catch it loudly instead of letting it degrade silently.
  /// </summary>
  unsafe void CheckTiming(MujocoLib.mjModel_* model) {
    double modelDt = model->opt.timestep;
    double policyDt = _policy.CtrlDt;
    double actualCtrlDt = modelDt * actionDecimation;

    if (Math.Abs(actualCtrlDt - policyDt) > 1e-6) {
      Debug.LogError(
          $"CreatureAgent: control rate mismatch. The policy trained at " +
          $"{policyDt * 1000.0:F1} ms per action ({1.0 / policyDt:F0} Hz) but this scene " +
          $"gives {actualCtrlDt * 1000.0:F1} ms ({1.0 / actualCtrlDt:F0} Hz): " +
          $"timestep {modelDt:F4} s x decimation {actionDecimation}. " +
          $"The plug-in ignores the MJCF timestep and uses Time.fixedDeltaTime, so set " +
          $"Project Settings > Time > Fixed Timestep to {policyDt / actionDecimation:F4}. " +
          $"The creature will move poorly until you do.", this);
    } else if (logBindings) {
      Debug.Log($"CreatureAgent: timing OK -- {modelDt:F4} s x {actionDecimation} = " +
                $"{1.0 / actualCtrlDt:F0} Hz control, matching training.", this);
    }
  }

  /// <summary>
  /// Writes the training start pose into mjData. Unity's importer drops
  /// &lt;keyframe&gt;, so the angles travel inside policy.json instead.
  /// </summary>
  unsafe void ApplyHomePose(MujocoLib.mjModel_* model, MujocoLib.mjData_* data) {
    var home = _policy.HomeJointPos;
    if (home == null) {
      Debug.LogWarning(
          "CreatureAgent: policy.json has no homeJointPos (exported by an older " +
          "export_policy.py). Starting from the imported pose instead, which may " +
          "not match training. Re-run export_policy.py to fix.", this);
      return;
    }
    for (int i = 0; i < joints.Count; i++) {
      data->qpos[_qposAddr[i]] = home[i];
      data->qvel[_dofAddr[i]] = 0.0;
    }
    MujocoLib.mj_forward(model, data);
    if (logBindings) {
      Debug.Log("CreatureAgent: applied home pose from policy.json.", this);
    }
  }

  unsafe void BuildObservation(MujocoLib.mjData_* data) {
    // xmat is row-major 3x3 per body: R[r][c] = xmat[9*id + 3*r + c], torso -> world.
    double* xmat = data->xmat + 9 * _torsoId;
    double* qpos = data->qpos;
    double* qvel = data->qvel;

    // gravity_local = R^T * (0,0,-1)  ->  -R[2][c]
    _obs[0] = (float)(-xmat[6]);
    _obs[1] = (float)(-xmat[7]);
    _obs[2] = (float)(-xmat[8]);

    // linvel_local = R^T * qvel[0:3]  ->  sum over r of R[r][c] * qvel[r]
    for (int c = 0; c < 3; c++) {
      _obs[3 + c] =
          (float)(xmat[c] * qvel[0] + xmat[3 + c] * qvel[1] + xmat[6 + c] * qvel[2]);
    }

    // angvel_local: a MuJoCo free joint already stores this in the body frame.
    _obs[6] = (float)qvel[3];
    _obs[7] = (float)qvel[4];
    _obs[8] = (float)qvel[5];

    int n = joints.Count;
    for (int i = 0; i < n; i++) {
      _obs[9 + i] = (float)qpos[_qposAddr[i]];
      _obs[9 + n + i] = (float)qvel[_dofAddr[i]];
    }
    Array.Copy(_lastAction, 0, _obs, 9 + 2 * n, _lastAction.Length);
  }
}

}  // namespace Creature
