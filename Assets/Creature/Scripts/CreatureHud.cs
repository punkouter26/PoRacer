using System;
using Mujoco;
using UnityEngine;

namespace Creature {

/// <summary>
/// On-screen readout for verifying the policy in Play mode.
///
/// Values are read straight out of mjData in MuJoCo's own frame and units, so
/// they are directly comparable with what training reported and with
/// mjx_training/view_creature.py -- not Unity transforms, which go through an
/// axis conversion.
///
/// Reference figures for the walk03 policy, measured in native MuJoCo:
///   speed ~1.1 m/s over the first 4 s, torso height ~0.31, never below 0.16.
/// </summary>
public class CreatureHud : MonoBehaviour {

  [Tooltip("Seconds of history used for the smoothed speed readout.")]
  public float speedWindow = 0.5f;

  [Tooltip("Episode terminates below this height during training.")]
  public float fallHeight = 0.16f;

  bool _ready;
  double _startX;
  double _lastX;
  float _lastSampleTime;
  float _speed;
  float _elapsed;
  double _minHeight = double.MaxValue;
  bool _everFell;

  GUIStyle _box, _label;

  unsafe void Update() {
    if (!MjScene.InstanceExists || MjScene.Instance.Data == null) {
      return;
    }
    var d = MjScene.Instance.Data;
    double x = d->qpos[0];
    double h = d->qpos[2];

    if (!_ready) {
      _ready = true;
      _startX = x;
      _lastX = x;
      _lastSampleTime = Time.time;
      return;
    }

    _elapsed += Time.deltaTime;
    _minHeight = Math.Min(_minHeight, h);
    if (h < fallHeight) {
      _everFell = true;
    }

    float dt = Time.time - _lastSampleTime;
    if (dt >= speedWindow) {
      _speed = (float)((x - _lastX) / dt);
      _lastX = x;
      _lastSampleTime = Time.time;
    }
  }

  unsafe void OnGUI() {
    if (!_ready || !MjScene.InstanceExists || MjScene.Instance.Data == null) {
      return;
    }
    if (_box == null) {
      _box = new GUIStyle(GUI.skin.box) { alignment = TextAnchor.UpperLeft, padding = new RectOffset(12, 12, 10, 10) };
      _label = new GUIStyle(GUI.skin.label) { fontSize = 14, richText = true };
    }

    var d = MjScene.Instance.Data;
    double x = d->qpos[0], h = d->qpos[2];
    double travelled = x - _startX;

    string ok = _everFell
        ? "<color=#ff6b6b>FELL</color>"
        : "<color=#5ce65c>UPRIGHT</color>";

    GUILayout.BeginArea(new Rect(12, 12, 320, 190), GUIContent.none, _box);
    GUILayout.Label("<b>Creature verification</b>", _label);
    GUILayout.Label($"status      {ok}", _label);
    GUILayout.Label($"speed       {_speed:F2} m/s", _label);
    GUILayout.Label($"travelled   {travelled:F2} m in {_elapsed:F1} s", _label);
    GUILayout.Label($"avg speed   {(_elapsed > 0.01f ? travelled / _elapsed : 0):F2} m/s", _label);
    GUILayout.Label($"height      {h:F3}  (min {_minHeight:F3})", _label);
    GUILayout.Label($"<color=#9aa4b0>expect ~1.1 m/s, height &gt; {fallHeight:F2}</color>", _label);
    GUILayout.EndArea();
  }
}

}  // namespace Creature
