using Mujoco;
using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// Cross-simulator collision stand-ins (AGENTS rule M: creatures never pass through each
    /// other). MuJoCo and PhysX are separate solvers, so every worm gets a copy of its five
    /// segment capsules in the other simulator:
    ///
    ///   * MuJoCo side: five mocap bodies per PhysX worm, following its segments. They must
    ///     exist before MjScene compiles, which is why they are built in the spawn frame.
    ///   * PhysX side: five kinematic capsules per MuJoCo worm, following its segments.
    ///
    /// Built for every PhysX worm and every MuJoCo worm, so every MuJoCo/PhysX pair is
    /// covered whatever the lane count. Worms in the same simulator need none: MuJoCo worms
    /// share one MjScene and PhysX worms share the PhysX scene. Stand-ins never touch each
    /// other in a way that matters (kinematic-kinematic pairs make no PhysX contacts, and
    /// MuJoCo skips contacts between two mocap bodies).
    ///
    /// Both are one-way (each simulator sees the other worm as an immovable moving body)
    /// and one physics step behind. With the lanes 2 m apart a contact is an accident, not
    /// a race mechanic; the stand-ins exist so that an accident resolves as a collision
    /// rather than two worms ghosting through each other.
    /// </summary>
    internal static class WormProxyBuilder
    {
        public static void BuildMocapProxies(Transform mujocoWorld, Transform[] physxSegments,
                                             WormRig rig, int lane)
        {
            for (int segmentIndex = 0; segmentIndex < physxSegments.Length; segmentIndex++)
            {
                Transform source = physxSegments[segmentIndex];
                var mocapObject = new GameObject($"MocapProxy_L{lane}_Seg{segmentIndex}");
                mocapObject.transform.SetParent(mujocoWorld, false);
                mocapObject.transform.SetPositionAndRotation(source.position, source.rotation);
                mocapObject.AddComponent<MjMocapBody>();

                var geomObject = new GameObject("geom");
                geomObject.transform.SetParent(mocapObject.transform, false);
                // PhysX segments run along local Z; MjCapsuleShape runs along local Y.
                geomObject.transform.localRotation = Quaternion.FromToRotation(Vector3.up, Vector3.forward);
                MjGeom geom = geomObject.AddComponent<MjGeom>();
                geom.ShapeType = MjShapeComponent.ShapeTypes.Capsule;
                geom.Capsule.Radius = rig.SegmentRadius;
                geom.Capsule.HalfHeight = rig.SegmentHalfLength;
                MujocoWorldBuilder.ApplyContact(geom, rig.Friction);

                mocapObject.AddComponent<MocapProxyFollower>().Bind(source);
            }
        }

        public static GameObject BuildKinematicProxies(Transform[] mujocoSegmentGeoms, WormRig rig,
                                                       PhysicsMaterial physicsMaterial, int lane)
        {
            var root = new GameObject($"KinematicProxies_L{lane}");
            for (int segmentIndex = 0; segmentIndex < mujocoSegmentGeoms.Length; segmentIndex++)
            {
                Transform source = mujocoSegmentGeoms[segmentIndex];
                var proxyObject = new GameObject($"KinematicProxy_L{lane}_Seg{segmentIndex}");
                proxyObject.transform.SetParent(root.transform, false);
                proxyObject.transform.SetPositionAndRotation(source.position, source.rotation);

                Rigidbody body = proxyObject.AddComponent<Rigidbody>();
                body.isKinematic = true;
                body.useGravity = false;
                body.interpolation = RigidbodyInterpolation.None;

                // The MjGeom transform it follows carries the capsule along local Y.
                CapsuleCollider capsule = proxyObject.AddComponent<CapsuleCollider>();
                capsule.direction = 1;
                capsule.radius = rig.SegmentRadius;
                capsule.height = 2f * (rig.SegmentHalfLength + rig.SegmentRadius);
                capsule.center = Vector3.zero;
                capsule.sharedMaterial = physicsMaterial;

                proxyObject.AddComponent<KinematicProxyFollower>().Bind(source);
            }
            return root;
        }
    }
}
