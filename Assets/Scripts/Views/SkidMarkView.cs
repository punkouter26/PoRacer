using PoRacer.Presentation;
using PoRacer.Systems;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Ground trail behind a racer: a flat TrailRenderer ribbon pinned to the
    /// track surface under the creature's root, emitting only while it actually
    /// moves near the ground. The marks fade out over a few seconds, so the
    /// track shows where the race has been. Added by Systems_Spawn.
    /// </summary>
    public sealed class SkidMarkView : MonoBehaviour
    {
        private const float TRAIL_SECONDS = 7f;
        private const float MIN_EMIT_SPEED = 0.25f;
        private const float MAX_GROUND_CLEARANCE = 0.9f;
        private const float SURFACE_LIFT = 0.05f;

        private static readonly Color MarkColor = new(0.22f, 0.16f, 0.1f, 0.35f);

        private TrackKind _kind;
        private Transform _root;
        private Transform _skid;
        private TrailRenderer _trail;
        private Vector3 _lastPosition;

        public void Initialize(TrackKind kind)
        {
            _kind = kind;
        }

        private void Start()
        {
            _root = transform;
            _lastPosition = _root.position;
            Material material = FxUtil.SoftParticleMaterial();
            if (material == null)
            {
                enabled = false;
                return;
            }
            var skid = new GameObject("Skid");
            // World-parented: the ribbon must not inherit the creature's tumbling.
            skid.transform.SetParent(null, false);
            // Lying flat: with TransformZ alignment the ribbon spans the XZ plane.
            skid.transform.rotation = Quaternion.Euler(90f, 0f, 0f);
            _trail = skid.AddComponent<TrailRenderer>();
            _trail.time = TRAIL_SECONDS;
            _trail.startWidth = 0.35f;
            _trail.endWidth = 0.05f;
            _trail.minVertexDistance = 0.25f;
            _trail.alignment = LineAlignment.TransformZ;
            _trail.startColor = MarkColor;
            _trail.endColor = new Color(MarkColor.r, MarkColor.g, MarkColor.b, 0f);
            _trail.sharedMaterial = material;
            _trail.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            _trail.receiveShadows = false;
            _trail.emitting = false;
            _skid = skid.transform;
        }

        private void LateUpdate()
        {
            if (_skid == null)
            {
                return;
            }
            Vector3 position = _root.position;
            float speed = (position - _lastPosition).magnitude / Mathf.Max(Time.deltaTime, 0.0001f);
            _lastPosition = position;
            float surfaceY = Systems_TrackBuilder.SurfaceHeight(_kind, position.x, position.z);
            bool grounded = position.y - surfaceY < MAX_GROUND_CLEARANCE;
            _trail.emitting = grounded && speed > MIN_EMIT_SPEED;
            _skid.position = new Vector3(position.x, surfaceY + SURFACE_LIFT, position.z);
        }

        private void OnDestroy()
        {
            // The trail lives outside the racer hierarchy; despawn must reap it.
            if (_skid != null)
            {
                Destroy(_skid.gameObject);
            }
        }
    }
}
