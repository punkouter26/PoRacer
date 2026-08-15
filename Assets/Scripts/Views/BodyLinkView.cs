using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Visual connective tissue: a stretched capsule between every articulation
    /// body part and its parent part, so creatures read as one solid animal
    /// instead of floating pieces. Purely cosmetic — no colliders, no physics,
    /// transforms follow the simulated parts in LateUpdate. Link renderers share
    /// the creature's material so racer tinting and batching keep working.
    /// </summary>
    public sealed class BodyLinkView : MonoBehaviour
    {
        private const float MIN_LINK_DISTANCE = 0.02f;
        private const float THICKNESS_SCALE = 0.55f;
        private const float MIN_THICKNESS = 0.05f;
        private const float MAX_THICKNESS = 0.3f;

        private Transform[] _links;
        private Transform[] _parents;
        private Transform[] _children;
        private float[] _thickness;

        private void Awake()
        {
            ArticulationBody[] bodies = GetComponentsInChildren<ArticulationBody>();
            Material creatureMaterial = null;
            Renderer firstRenderer = GetComponentInChildren<Renderer>();
            if (firstRenderer != null)
            {
                creatureMaterial = firstRenderer.sharedMaterial;
            }

            var links = new System.Collections.Generic.List<Transform>();
            var parents = new System.Collections.Generic.List<Transform>();
            var children = new System.Collections.Generic.List<Transform>();
            var thickness = new System.Collections.Generic.List<float>();
            for (int bodyIndex = 0; bodyIndex < bodies.Length; bodyIndex++)
            {
                ArticulationBody body = bodies[bodyIndex];
                if (body.isRoot)
                {
                    continue;
                }
                ArticulationBody parentBody = body.transform.parent != null
                    ? body.transform.parent.GetComponentInParent<ArticulationBody>()
                    : null;
                if (parentBody == null)
                {
                    continue;
                }

                var link = GameObject.CreatePrimitive(PrimitiveType.Capsule);
                link.name = "BodyLink";
                // Disable before the deferred Destroy so the collider can never
                // touch the physics world, not even for one frame.
                var linkCollider = link.GetComponent<Collider>();
                linkCollider.enabled = false;
                Destroy(linkCollider);
                link.transform.SetParent(transform, false);
                // Invisible until the first LateUpdate poses it.
                link.transform.localScale = Vector3.zero;
                var linkRenderer = link.GetComponent<MeshRenderer>();
                if (creatureMaterial != null)
                {
                    linkRenderer.sharedMaterial = creatureMaterial;
                }

                // Limb girth follows the smaller of the two connected parts.
                float childSize = EstimateRadius(body.transform);
                float parentSize = EstimateRadius(parentBody.transform);
                float radius = Mathf.Clamp(
                    Mathf.Min(childSize, parentSize) * THICKNESS_SCALE, MIN_THICKNESS, MAX_THICKNESS);

                links.Add(link.transform);
                parents.Add(parentBody.transform);
                children.Add(body.transform);
                thickness.Add(radius);
            }
            _links = links.ToArray();
            _parents = parents.ToArray();
            _children = children.ToArray();
            _thickness = thickness.ToArray();
        }

        private void LateUpdate()
        {
            for (int linkIndex = 0; linkIndex < _links.Length; linkIndex++)
            {
                Transform parent = _parents[linkIndex];
                Transform child = _children[linkIndex];
                if (parent == null || child == null)
                {
                    continue;
                }
                Vector3 from = parent.position;
                Vector3 to = child.position;
                Vector3 delta = to - from;
                float distance = delta.magnitude;
                Transform link = _links[linkIndex];
                if (distance < MIN_LINK_DISTANCE)
                {
                    link.localScale = Vector3.zero;
                    continue;
                }
                float radius = _thickness[linkIndex];
                link.position = (from + to) * 0.5f;
                link.rotation = Quaternion.FromToRotation(Vector3.up, delta / distance);
                // A capsule primitive is 2 units tall at scale 1, so scale.y is
                // half the span; a little overshoot tucks the caps into the parts.
                link.localScale = new Vector3(radius, distance * 0.5f + radius * 0.5f, radius);
            }
        }

        private static float EstimateRadius(Transform part)
        {
            var partRenderer = part.GetComponent<Renderer>();
            if (partRenderer == null)
            {
                partRenderer = part.GetComponentInChildren<Renderer>();
            }
            if (partRenderer == null)
            {
                return MIN_THICKNESS;
            }
            Vector3 size = partRenderer.bounds.size;
            return Mathf.Min(size.x, Mathf.Min(size.y, size.z)) * 0.5f;
        }
    }
}
