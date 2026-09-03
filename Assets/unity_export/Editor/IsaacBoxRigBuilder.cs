using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

using IsaacBox;
using PoRacer.IsaacPorts;

namespace IsaacBox.EditorTools
{
    /// <summary>
    /// Builds the IsaacBox ArticulationBody hierarchy from <see cref="IsaacBoxRigAsset"/> and hangs
    /// the authored skinned mesh on it.
    ///
    /// Two-stage, deliberately: the articulation is built FIRST from the rig JSON as clean
    /// world-aligned empties (the same recipe as the IsaacH1 builder, so the frame map is
    /// applied here and only here). THEN the imported FBX is instantiated, aligned, and its
    /// bones are re-parented under the links that own them. The skinned-mesh renderers only
    /// care where their bone Transforms are, so this is all the "rig import" there is; no
    /// ArticulationBody ever sits on a bone with an authored rest rotation, and no joint
    /// anchor has to compensate for one.
    ///
    /// The FBX is not trusted: every bone's world position is checked against the rig
    /// JSON (which came from the GLB twin) to 5 mm, after trying the four yaw alignments a
    /// model importer can plausibly produce. A mismatch aborts the build with the numbers.
    /// </summary>
    public static class IsaacBoxRigBuilder
    {
        public const string RootName = "IsaacBox";
        public const float BonePositionGateM = 0.005f;

        public static GameObject Build(IsaacBoxRigAsset rig, GameObject fbxModel, out string report)
        {
            if (rig == null) throw new ArgumentNullException(nameof(rig));
            var sb = new System.Text.StringBuilder();

            var root = new GameObject(RootName);
            var created = new Dictionary<string, GameObject>(rig.bodies.Length);

            // bodies are stored breadth-first, so one forward pass always sees the parent first
            foreach (var def in rig.bodies)
            {
                GameObject parentGo;
                if (def.isRoot)
                {
                    parentGo = root;
                }
                else if (!created.TryGetValue(def.parent, out parentGo))
                {
                    UnityEngine.Object.DestroyImmediate(root);
                    throw new InvalidOperationException(
                        $"body '{def.name}' names parent '{def.parent}', which has not been created yet");
                }

                var go = new GameObject(def.name);
                go.transform.SetParent(parentGo.transform, false);
                go.transform.localPosition = IsaacFrameMap.Pos(def.localPosIsaac);
                Vector4 q = def.localRotIsaacWxyz;
                go.transform.localRotation = IsaacFrameMap.RotFromWxyz(q.x, q.y, q.z, q.w);
                go.transform.localScale = Vector3.one;

                var body = go.AddComponent<ArticulationBody>();
                ConfigureBody(body, def, rig);
                AddColliders(go, def, rig);
                created[def.name] = go;
            }
            sb.Append($"articulation: {created.Count} links\n");

            if (fbxModel != null)
            {
                AttachSkin(root, rig, created, fbxModel, sb);
            }
            else
            {
                sb.Append("no FBX supplied: physics-only rig (no skinned mesh)\n");
            }

            report = sb.ToString();
            return root;
        }

        static void ConfigureBody(ArticulationBody body, BoyBodyDef def, IsaacBoxRigAsset rig)
        {
            var p = rig.physics;

            body.mass = def.mass;
            body.centerOfMass = IsaacFrameMap.Pos(def.comIsaac);
            body.inertiaTensor = IsaacFrameMap.InertiaDiagToUnity(def.inertiaDiagIsaac);
            body.inertiaTensorRotation = Quaternion.identity;

            body.linearDamping = p.linearDamping;
            body.angularDamping = p.angularDamping;
            body.jointFriction = p.jointFriction;
            body.maxLinearVelocity = p.maxLinearVelocity;
            body.maxAngularVelocity = p.maxAngularVelocity;
            body.maxDepenetrationVelocity = p.maxDepenetrationVelocity;
            body.solverIterations = p.solverPositionIterations;
            body.solverVelocityIterations = p.solverVelocityIterations;
            body.useGravity = true;

            if (def.isRoot)
            {
                body.immovable = false;
                return;
            }

            var j = def.joint;
            Vector3 axisUnity = IsaacFrameMap.Axis(j.axisInChildIsaac).normalized;

            body.jointType = ArticulationJointType.RevoluteJoint;
            body.matchAnchors = true;
            body.anchorPosition = Vector3.zero;           // the joint sits on the link origin
            body.anchorRotation = IsaacFrameMap.AnchorRotationForAxis(axisUnity);

            body.twistLock = ArticulationDofLock.LimitedMotion;
            body.swingYLock = ArticulationDofLock.LockedMotion;
            body.swingZLock = ArticulationDofLock.LockedMotion;

            var d = body.xDrive;
            d.lowerLimit = j.lowerRad * Mathf.Rad2Deg;
            d.upperLimit = j.upperRad * Mathf.Rad2Deg;
            d.stiffness = j.stiffness;                 // radian convention, measured by rung 2b
            d.damping = j.damping;
            d.forceLimit = j.effortLimit;
            d.target = j.defaultPosRad * Mathf.Rad2Deg;
            d.targetVelocity = 0f;
            d.driveType = ArticulationDriveType.Force;
            body.xDrive = d;

            body.maxJointVelocity = p.maxAngularVelocity;
        }

        static void AddColliders(GameObject go, BoyBodyDef def, IsaacBoxRigAsset rig)
        {
            if (def.colliders == null) return;
            foreach (var c in def.colliders)
            {
                Vector3 center = IsaacFrameMap.Pos(c.centerIsaac);
                switch (c.kind)
                {
                    case "box":
                    {
                        var box = go.AddComponent<BoxCollider>();
                        box.center = center;
                        Vector3 s = IsaacFrameMap.Pos(c.sizeIsaac);
                        box.size = new Vector3(Mathf.Abs(s.x), Mathf.Abs(s.y), Mathf.Abs(s.z));
                        box.contactOffset = rig.physics.contactOffset;
                        break;
                    }
                    case "sphere":
                    {
                        var sph = go.AddComponent<SphereCollider>();
                        sph.center = center;
                        sph.radius = c.radius;
                        sph.contactOffset = rig.physics.contactOffset;
                        break;
                    }
                    case "capsule":
                    {
                        var cap = go.AddComponent<CapsuleCollider>();
                        cap.center = center;
                        cap.radius = c.radius;
                        // USD height is the cylinder part; Unity's includes both caps.
                        cap.height = c.height + 2f * c.radius;
                        // Isaac X -> Unity Z (2), Isaac Y -> Unity X (0), Isaac Z -> Unity Y (1)
                        cap.direction = c.axis == "X" ? 2 : c.axis == "Y" ? 0 : 1;
                        cap.contactOffset = rig.physics.contactOffset;
                        break;
                    }
                    default:
                        Debug.LogWarning($"[IsaacBox] unknown collider kind '{c.kind}' on {def.name}; skipped.");
                        break;
                }
            }
        }

        // ------------------------------------------------------------------ skin --
        static void AttachSkin(GameObject root, IsaacBoxRigAsset rig, Dictionary<string, GameObject> links,
                               GameObject fbxModel, System.Text.StringBuilder sb)
        {
            var instance = (GameObject)PrefabUtility.InstantiatePrefab(fbxModel);
            PrefabUtility.UnpackPrefabInstance(instance, PrefabUnpackMode.Completely, InteractionMode.AutomatedAction);
            instance.name = "Skin";
            instance.transform.SetParent(root.transform, false);
            // The container's origin IS the hips link (the root link has localPos 0, as on the
            // H1, so a spawn position is a hips position). The FBX is authored with its feet
            // at the origin and the hips 0.76 m up, so shift it down by the hips rest position
            // before comparing or attaching anything.
            Vector3 rootOffset = IsaacFrameMap.Pos(RootWorldPosIsaac(rig));
            instance.transform.localPosition = -rootOffset;
            instance.transform.localRotation = Quaternion.identity;
            instance.transform.localScale = Vector3.one;

            // an Animator/Animation would fight the physics every frame
            foreach (var a in instance.GetComponentsInChildren<Animator>(true)) UnityEngine.Object.DestroyImmediate(a);
            foreach (var a in instance.GetComponentsInChildren<Animation>(true)) UnityEngine.Object.DestroyImmediate(a);

            var bones = new Dictionary<string, Transform>();
            foreach (var t in instance.GetComponentsInChildren<Transform>(true))
                if (!bones.ContainsKey(t.name)) bones[t.name] = t;

            // which yaw does this importer give the model? Try the four candidates and keep
            // the one that puts every articulated bone on its rig position.
            float bestErr = float.MaxValue;
            float bestYaw = 0f;
            string bestWhere = "";
            foreach (float yaw in new[] { 0f, 180f, 90f, 270f })
            {
                // yaw about the character's own vertical axis, which passes through the feet origin
                instance.transform.localRotation = Quaternion.Euler(0f, yaw, 0f);
                instance.transform.localPosition = Quaternion.Euler(0f, yaw, 0f) * -rootOffset;
                float err = WorstBoneError(rig, bones, root.transform, out string where);
                if (err < bestErr) { bestErr = err; bestYaw = yaw; bestWhere = where; }
            }
            instance.transform.localRotation = Quaternion.Euler(0f, bestYaw, 0f);
            instance.transform.localPosition = Quaternion.Euler(0f, bestYaw, 0f) * -rootOffset;

            float hipsY = bones.TryGetValue("hips", out var hipsBone)
                ? root.transform.InverseTransformPoint(hipsBone.position).y + rootOffset.y
                : float.NaN;
            sb.Append($"skin: FBX yaw {bestYaw:F0} deg, worst bone offset {bestErr * 1000f:F2} mm at {bestWhere}, " +
                      $"hips bone authored at y = {hipsY:F4} m (rig {rootOffset.y:F4})\n");

            if (bestErr > BonePositionGateM)
            {
                UnityEngine.Object.DestroyImmediate(instance);
                throw new InvalidOperationException(
                    $"IsaacBox_Character.fbx bones do not sit on the rig positions from isaacbox_rig.json: worst " +
                    $"{bestErr * 1000f:F1} mm at {bestWhere} after trying yaws 0/90/180/270 (gate " +
                    $"{BonePositionGateM * 1000f:F0} mm). Hips bone y = {hipsY:F4} m; a value near 76 or " +
                    $"0.0076 means the FBX import scale is off (Model tab > Scale Factor / Convert Units). " +
                    $"The GLB and the FBX must describe the same rest pose.");
            }

            // hang every articulated bone under its link; the rest of the skeleton rides along
            int attached = 0;
            foreach (var def in rig.bodies)
            {
                if (string.IsNullOrEmpty(def.boneName)) continue;
                if (!bones.TryGetValue(def.boneName, out var bone))
                {
                    UnityEngine.Object.DestroyImmediate(instance);
                    throw new InvalidOperationException($"FBX has no bone named '{def.boneName}' for link {def.name}");
                }
                bone.SetParent(links[def.name].transform, true);
                attached++;
            }

            // the renderers can live anywhere; they only need their bones. Keep them together.
            var skinHolder = new GameObject("Visual");
            skinHolder.transform.SetParent(root.transform, false);
            int renderers = 0, verts = 0;
            foreach (var smr in instance.GetComponentsInChildren<SkinnedMeshRenderer>(true))
            {
                smr.transform.SetParent(skinHolder.transform, true);
                if (hipsBone != null) smr.rootBone = hipsBone;
                smr.updateWhenOffscreen = true;   // bounds follow the articulation, not the import pose
                renderers++;
                if (smr.sharedMesh != null) verts += smr.sharedMesh.vertexCount;
            }
            foreach (var mr in instance.GetComponentsInChildren<MeshRenderer>(true))
                mr.transform.SetParent(skinHolder.transform, true);

            // whatever is left of the import (the armature holder) is empty now
            UnityEngine.Object.DestroyImmediate(instance);
            sb.Append($"skin: {attached} bones attached to links, {renderers} skinned renderers, {verts:N0} vertices\n");
        }

        static Vector3 RootWorldPosIsaac(IsaacBoxRigAsset rig)
        {
            foreach (var def in rig.bodies)
                if (def.isRoot) return def.worldPosIsaac;
            throw new InvalidOperationException("rig has no root body");
        }

        /// <summary>
        /// Worst distance between an articulated bone and its link's rest position, both
        /// expressed in the container's frame whose origin is the hips link.
        /// </summary>
        static float WorstBoneError(IsaacBoxRigAsset rig, Dictionary<string, Transform> bones, Transform container,
                                    out string where)
        {
            float worst = 0f;
            where = "";
            Vector3 rootIsaac = RootWorldPosIsaac(rig);
            foreach (var def in rig.bodies)
            {
                if (string.IsNullOrEmpty(def.boneName)) continue;
                if (!bones.TryGetValue(def.boneName, out var bone)) { where = def.boneName + " (missing)"; return float.MaxValue; }
                Vector3 expected = IsaacFrameMap.Pos(def.worldPosIsaac - rootIsaac);
                Vector3 got = container.InverseTransformPoint(bone.position);
                float e = (got - expected).magnitude;
                if (e > worst) { worst = e; where = def.boneName; }
            }
            return worst;
        }
    }
}
