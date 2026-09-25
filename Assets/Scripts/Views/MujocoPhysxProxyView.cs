using System.Collections.Generic;
using Mujoco;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Gives a MuJoCo racer a body in PhysX (AGENTS rule M). MuJoCo racers have no PhysX
    /// colliders at all - MojucuBoy's prefab carries none - so fruit and the PhysX racers
    /// passed straight through him. This mirrors every one of his MjGeoms as a kinematic
    /// PhysX collider of the same shape and size, moved onto the geom every physics step.
    ///
    /// One-way, like the worm race's stand-ins: PhysX sees him as an immovable moving body
    /// (fruit bounces off him, a PhysX racer is shoved), and the proxies are one step
    /// behind MuJoCo. The other direction is Systems_MujocoWorld's fruit pool.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class MujocoPhysxProxyView : MonoBehaviour
    {
        private readonly List<Transform> _sources = new();
        private readonly List<Rigidbody> _proxies = new();
        private GameObject _root;

        private void Start()
        {
            _root = new GameObject($"PhysxProxies_{name}");
            MjGeom[] geoms = GetComponentsInChildren<MjGeom>();
            for (int geomIndex = 0; geomIndex < geoms.Length; geomIndex++)
            {
                MjGeom geom = geoms[geomIndex];
                var proxy = new GameObject(geom.name);
                proxy.transform.SetParent(_root.transform, false);
                proxy.transform.SetPositionAndRotation(geom.transform.position, geom.transform.rotation);
                if (!AddCollider(proxy, geom))
                {
                    Destroy(proxy);
                    continue;
                }
                Rigidbody body = proxy.AddComponent<Rigidbody>();
                body.isKinematic = true;
                body.useGravity = false;
                body.interpolation = RigidbodyInterpolation.None;
                _sources.Add(geom.transform);
                _proxies.Add(body);
            }
        }

        private void FixedUpdate()
        {
            for (int proxyIndex = 0; proxyIndex < _proxies.Count; proxyIndex++)
            {
                Transform source = _sources[proxyIndex];
                _proxies[proxyIndex].MovePosition(source.position);
                _proxies[proxyIndex].MoveRotation(source.rotation);
            }
        }

        private void OnDestroy()
        {
            if (_root != null)
            {
                Destroy(_root);
            }
        }

        /// <summary>The PhysX twin of a geom: MuJoCo sizes are half-sizes, capsules run along local Y.</summary>
        private static bool AddCollider(GameObject proxy, MjGeom geom)
        {
            switch (geom.ShapeType)
            {
                case MjShapeComponent.ShapeTypes.Capsule:
                {
                    var capsule = proxy.AddComponent<CapsuleCollider>();
                    capsule.direction = 1;
                    capsule.radius = geom.Capsule.Radius;
                    capsule.height = 2f * (geom.Capsule.HalfHeight + geom.Capsule.Radius);
                    return true;
                }
                case MjShapeComponent.ShapeTypes.Cylinder:
                {
                    var capsule = proxy.AddComponent<CapsuleCollider>();
                    capsule.direction = 1;
                    capsule.radius = geom.Cylinder.Radius;
                    capsule.height = 2f * Mathf.Max(geom.Cylinder.HalfHeight, geom.Cylinder.Radius);
                    return true;
                }
                case MjShapeComponent.ShapeTypes.Box:
                {
                    var box = proxy.AddComponent<BoxCollider>();
                    box.size = 2f * geom.Box.Extents;
                    return true;
                }
                case MjShapeComponent.ShapeTypes.Sphere:
                {
                    var sphere = proxy.AddComponent<SphereCollider>();
                    sphere.radius = geom.Sphere.Radius;
                    return true;
                }
                case MjShapeComponent.ShapeTypes.Ellipsoid:
                {
                    var box = proxy.AddComponent<BoxCollider>();
                    box.size = 2f * geom.Ellipsoid.Radiuses;
                    return true;
                }
                default:
                    return false;
            }
        }
    }
}
