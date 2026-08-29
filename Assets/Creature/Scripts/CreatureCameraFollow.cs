using UnityEngine;

namespace Creature {

/// <summary>
/// Keeps the camera on the creature. It covers ~1.4 m/s, so a fixed camera
/// loses it within a couple of seconds.
/// </summary>
public class CreatureCameraFollow : MonoBehaviour {

  [Tooltip("Usually the imported 'torso' GameObject.")]
  public Transform target;

  [Tooltip("Offset from the target, in world space.")]
  public Vector3 offset = new Vector3(-2.2f, 1.3f, 3.2f);

  [Tooltip("0 = rigid, higher = smoother trailing.")]
  public float smoothing = 4f;

  [Tooltip("Follow only the ground plane, so the view does not bob with the gait.")]
  public bool ignoreVerticalMotion = true;

  Vector3 _velocity;
  float _lockedY;
  bool _haveLockedY;

  void LateUpdate() {
    if (target == null) {
      return;
    }
    var focus = target.position;
    if (ignoreVerticalMotion) {
      if (!_haveLockedY) {
        _lockedY = focus.y;
        _haveLockedY = true;
      }
      focus.y = _lockedY;
    }

    var desired = focus + offset;
    transform.position = smoothing > 0f
        ? Vector3.SmoothDamp(transform.position, desired, ref _velocity, 1f / smoothing)
        : desired;
    transform.LookAt(focus);
  }
}

}  // namespace Creature
