using PoRacer.Presentation;
using PoRacer.Systems;
using UnityEngine;
using UnityEngine.Rendering;

namespace PoRacer.Views
{
    /// <summary>
    /// A soft dark blob on the ground under a racer, which is what stops a
    /// creature reading as floating when the real shadow cannot do it: the
    /// phone tier's shadow map is low-resolution and hard-edged, and the lowest
    /// quality tier turns it off. Shown only when FxBudget.ContactShadows says
    /// so; on the PC tier the soft shadow map and ambient occlusion already do
    /// this job and a blob would double-darken the ground.
    ///
    /// Sized once from the creature's spawn-pose bounds. It shrinks and fades as
    /// the body's lowest point rises off the ground, so a jump or a tumble
    /// lifts off its shadow instead of dragging a full-size blob along.
    ///
    /// Every blob is the same shared quad and shared material, so the whole
    /// field batches. Not shown on authored courses, whose ground is arbitrary
    /// geometry the analytic surface height does not describe. Added by
    /// Systems_Spawn.
    /// </summary>
    public sealed class ContactShadowView : MonoBehaviour
    {
        private const float MIN_SIZE = 0.4f;
        private const float MAX_SIZE = 3f;
        // Blob is a little wider than the body, like a real soft contact shadow.
        private const float SIZE_MARGIN = 1.15f;
        // Height of the body's lowest point above the ground at which the blob has gone.
        private const float FADE_HEIGHT = 1.4f;
        private const float SURFACE_LIFT = 0.02f;

        private TrackKind _kind;
        private Transform _root;
        private Transform _blob;
        private MeshRenderer _renderer;
        private float _size;
        // Root height above its own lowest point, measured in the spawn pose.
        private float _rootToFeet;

        public void Initialize(TrackKind kind)
        {
            _kind = kind;
        }

        private void Start()
        {
            _root = transform;
            Material material = FxUtil.ContactShadowMaterial();
            if (material == null || _kind.IsCourse())
            {
                enabled = false;
                return;
            }
            MeasureBody();

            var blob = new GameObject("ContactShadow");
            // World-parented: the blob must not inherit the creature's tumbling.
            blob.transform.SetParent(null, false);
            blob.AddComponent<MeshFilter>().sharedMesh = FxUtil.FlatQuad();
            _renderer = blob.AddComponent<MeshRenderer>();
            _renderer.sharedMaterial = material;
            _renderer.shadowCastingMode = ShadowCastingMode.Off;
            _renderer.receiveShadows = false;
            _renderer.lightProbeUsage = LightProbeUsage.Off;
            _renderer.reflectionProbeUsage = ReflectionProbeUsage.Off;
            _renderer.enabled = FxBudget.ContactShadows;
            _blob = blob.transform;
        }

        private void LateUpdate()
        {
            if (_blob == null)
            {
                return;
            }
            bool wanted = FxBudget.ContactShadows;
            if (_renderer.enabled != wanted)
            {
                _renderer.enabled = wanted;
            }
            if (!wanted)
            {
                return;
            }
            Vector3 position = _root.position;
            float surfaceY = Systems_TrackBuilder.SurfaceHeight(_kind, position.x, position.z);
            float gap = position.y - _rootToFeet - surfaceY;
            float presence = 1f - Mathf.Clamp01(gap / FADE_HEIGHT);
            _blob.position = new Vector3(position.x, surfaceY + SURFACE_LIFT, position.z);
            // Shrinking stands in for fading: the blob shares one material with the
            // whole field, and a per-racer alpha would need a property block that
            // takes it off the batched path.
            float scale = _size * Mathf.Lerp(0.35f, 1f, presence);
            _blob.localScale = new Vector3(scale, 1f, scale);
        }

        private void OnDestroy()
        {
            // The blob lives outside the racer hierarchy; despawn must reap it.
            if (_blob != null)
            {
                Destroy(_blob.gameObject);
            }
        }

        /// <summary>
        /// Footprint size and foot height from the renderers, once at spawn. The
        /// allocation from GetComponentsInChildren never reaches a frame loop.
        /// </summary>
        private void MeasureBody()
        {
            Renderer[] renderers = GetComponentsInChildren<Renderer>();
            if (renderers.Length == 0)
            {
                _size = MIN_SIZE;
                _rootToFeet = 0f;
                return;
            }
            Bounds bounds = renderers[0].bounds;
            for (int rendererIndex = 1; rendererIndex < renderers.Length; rendererIndex++)
            {
                bounds.Encapsulate(renderers[rendererIndex].bounds);
            }
            float footprint = Mathf.Max(bounds.size.x, bounds.size.z) * SIZE_MARGIN;
            _size = Mathf.Clamp(footprint, MIN_SIZE, MAX_SIZE);
            _rootToFeet = Mathf.Max(0f, _root.position.y - bounds.min.y);
        }
    }
}
