using System;
using UnityEngine;

namespace Creature.MojucuBoy
{
    /// <summary>
    /// Drives the authored character's skeleton from the MuJoCo bodies.
    ///
    /// MuJoCo simulates 13 bodies; the GLB rig has 19 bones. The six that carry no
    /// joint -- both clavicles, both hands, and the spine/chest and neck pairs that
    /// were fused -- are welded to whichever body owns them, so they follow rigidly
    /// rather than being left behind at the origin.
    ///
    /// Offsets are MEASURED at bind time, not assumed. build_mjcf.py authors every
    /// MuJoCo body frame to coincide with its bone's glTF bind frame, so most
    /// offsets come out as identity -- but capturing the actual relative transform
    /// costs nothing and means a change to the rig or a re-import cannot silently
    /// shear the character. Capture happens once, from the pose the hierarchy is in
    /// when this component initialises, so the character must be at its bind pose
    /// and aligned with the MJCF root at that moment.
    ///
    /// Runs in LateUpdate, after MjScene has written the physics results onto the
    /// Unity transforms in FixedUpdate.
    /// </summary>
    [DefaultExecutionOrder(200)]
    [DisallowMultipleComponent]
    public sealed class MojucuBoyVisualBinder : MonoBehaviour
    {
        [Serializable]
        public struct Link
        {
            [Tooltip("Bone name in the authored GLB skeleton.")]
            public string bone;

            [Tooltip("MuJoCo body GameObject name that drives it.")]
            public string body;
        }

        /// <summary>
        /// Default bone -> body map for Boy_Character_mujoco.glb. Six bones are
        /// welded: shoulder.L/R and hand.L/R carry no joint at all, and spine/chest
        /// and neck/head were fused into one body each when 19 bones were reduced to
        /// the 21 DOF that matches IsaacBox.
        /// </summary>
        public static readonly Link[] DefaultMap =
        {
            new Link { bone = "hips",        body = "hips" },
            new Link { bone = "spine",       body = "torso" },
            new Link { bone = "chest",       body = "torso" },
            new Link { bone = "neck",        body = "head" },
            new Link { bone = "head",        body = "head" },
            new Link { bone = "shoulder.L",  body = "torso" },
            new Link { bone = "upper_arm.L", body = "upper_arm_L" },
            new Link { bone = "forearm.L",   body = "forearm_L" },
            new Link { bone = "hand.L",      body = "forearm_L" },
            new Link { bone = "shoulder.R",  body = "torso" },
            new Link { bone = "upper_arm.R", body = "upper_arm_R" },
            new Link { bone = "forearm.R",   body = "forearm_R" },
            new Link { bone = "hand.R",      body = "forearm_R" },
            new Link { bone = "thigh.L",     body = "thigh_L" },
            new Link { bone = "shin.L",      body = "shin_L" },
            new Link { bone = "foot.L",      body = "foot_L" },
            new Link { bone = "thigh.R",     body = "thigh_R" },
            new Link { bone = "shin.R",      body = "shin_R" },
            new Link { bone = "foot.R",      body = "foot_R" },
        };

        [Tooltip("Root of the authored character (the GLB 'Rig' object).")]
        [SerializeField] private Transform _skinRoot;

        [Tooltip("Root of the imported MJCF hierarchy.")]
        [SerializeField] private Transform _physicsRoot;

        [SerializeField] private Link[] _map = DefaultMap;

        private Transform[] _bones;
        private Transform[] _bodies;
        private Vector3[] _offsetPosition;
        private Quaternion[] _offsetRotation;
        private bool _ready;

        private void Start() => Rebind();

        /// <summary>
        /// Capture the bind-pose offset of every bone relative to its driving body.
        /// Safe to call again after re-posing the hierarchy to its bind pose.
        /// </summary>
        public void Rebind()
        {
            _ready = false;
            if (_skinRoot == null || _physicsRoot == null)
            {
                Debug.LogError($"[{name}] skinRoot and physicsRoot must both be set.", this);
                return;
            }

            int count = _map.Length;
            _bones = new Transform[count];
            _bodies = new Transform[count];
            _offsetPosition = new Vector3[count];
            _offsetRotation = new Quaternion[count];

            for (int i = 0; i < count; i++)
            {
                _bones[i] = FindDeep(_skinRoot, _map[i].bone);
                _bodies[i] = FindDeep(_physicsRoot, _map[i].body);
                if (_bones[i] == null || _bodies[i] == null)
                {
                    Debug.LogError(
                        $"[{name}] cannot bind '{_map[i].bone}' -> '{_map[i].body}': "
                      + $"{(_bones[i] == null ? "bone" : "body")} not found. "
                      + "The character will not follow the physics.", this);
                    return;
                }

                // Offset that takes the body's frame to the bone's, captured now.
                Transform body = _bodies[i];
                Transform bone = _bones[i];
                _offsetPosition[i] = body.InverseTransformPoint(bone.position);
                _offsetRotation[i] = Quaternion.Inverse(body.rotation) * bone.rotation;
            }
            _ready = true;
        }

        private void LateUpdate()
        {
            if (!_ready)
            {
                return;
            }
            for (int i = 0; i < _bones.Length; i++)
            {
                Transform body = _bodies[i];
                // SetPositionAndRotation writes both in one call, avoiding the
                // redundant transform-hierarchy flush two separate setters cause.
                _bones[i].SetPositionAndRotation(
                    body.TransformPoint(_offsetPosition[i]),
                    body.rotation * _offsetRotation[i]);
            }
        }

        private static Transform FindDeep(Transform root, string wanted)
        {
            if (root.name == wanted)
            {
                return root;
            }
            for (int i = 0; i < root.childCount; i++)
            {
                Transform found = FindDeep(root.GetChild(i), wanted);
                if (found != null)
                {
                    return found;
                }
            }
            return null;
        }
    }
}
