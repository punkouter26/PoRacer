#if UNITY_EDITOR
using System.Collections.Generic;
using PoRacer.Agents;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEditor;
using UnityEngine;

namespace PoRacer.Editor
{
    /// <summary>
    /// Builds racer prefabs from the authored humanoid .glb models (Grandma,
    /// Grandpa, Matt, Nick). Unlike Editor_BuildCreatures, which invents a body
    /// out of primitives, this one hangs an ArticulationBody rig straight onto the
    /// imported skeleton: every driven bone keeps its own transform, so the
    /// SkinnedMeshRenderer deforms for free and the physics rig and the visible
    /// character can never drift apart.
    ///
    /// Eleven driven joints — torso lean, 3 per leg, 2 per arm — giving the same
    /// N*3+19 observation layout as the rest of the fleet.
    /// </summary>
    public static class Editor_BuildRiggedCreatures
    {
        private const int DECISION_PERIOD = 5;
        private const float TORQUE_PER_KG = 10f;
        // Ratios lifted from the tuned worm so every creature's drives feel alike.
        private const float STIFFNESS_RATIO = 5f;
        private const float DAMPING_RATIO = 1f / 3f;
        private const float JOINT_DRIVE_SCALE = 60f;
        // A joint never drives less than this share of the whole body: an ankle
        // moves a 1.5 kg foot but has to push 67 kg of racer off the ground, and
        // sizing its drive by the foot alone leaves it far too weak to walk.
        private const float MIN_DRIVEN_MASS_FRACTION = 0.25f;
        // Bone lengths are read in armature-local units (the .glb rigs are
        // authored in centimetres under a 0.01 armature scale).
        private const float LIMB_RADIUS_FRACTION = 0.17f;

        private static readonly string[] SourceModels =
        {
            "Assets/RIGGED_Grandma.glb",
            "Assets/RIGGED_Grandpa.glb",
            "Assets/RIGGED_Matt.glb",
            "Assets/RIGGED_Nick.glb"
        };

        /// <summary>One driven bone: who it is, what it weighs, how far it bends.</summary>
        private readonly struct JointSpec
        {
            public readonly string Bone;
            public readonly string ChildBone;
            public readonly float Mass;
            public readonly float LowerLimit;
            public readonly float UpperLimit;
            public readonly float GaitAmplitude;
            public readonly float GaitPhase;
            public readonly float GaitOffset;

            public JointSpec(string bone, string childBone, float mass, float lowerLimit, float upperLimit,
                float gaitAmplitude, float gaitPhase, float gaitOffset)
            {
                Bone = bone;
                ChildBone = childBone;
                Mass = mass;
                LowerLimit = lowerLimit;
                UpperLimit = upperLimit;
                GaitAmplitude = gaitAmplitude;
                GaitPhase = gaitPhase;
                GaitOffset = gaitOffset;
            }
        }

        private const float PELVIS_MASS = 12f;

        // Walk cycle: the legs run half a cycle apart, each knee leads its hip by a
        // quarter cycle so the foot clears the ground on the swing, and the arms
        // counter-swing against the leg on their own side. Knees and elbows carry a
        // negative DC offset so the limb rides bent instead of locked straight.
        private static readonly JointSpec[] Joints =
        {
            new("Spine02", "Spine01", 20f, -25f, 35f, 0.10f, 0f, 0.10f),

            new("LeftUpLeg", "LeftLeg", 8f, -75f, 45f, 0.55f, 0f, 0f),
            new("LeftLeg", "LeftFoot", 4f, -5f, 80f, 0.40f, Mathf.PI * 0.5f, -0.30f),
            new("LeftFoot", "LeftToeBase", 1.5f, -35f, 30f, 0.25f, Mathf.PI, 0f),

            new("RightUpLeg", "RightLeg", 8f, -75f, 45f, 0.55f, Mathf.PI, 0f),
            new("RightLeg", "RightFoot", 4f, -5f, 80f, 0.40f, Mathf.PI * 1.5f, -0.30f),
            new("RightFoot", "RightToeBase", 1.5f, -35f, 30f, 0.25f, 0f, 0f),

            new("LeftArm", "LeftForeArm", 2.5f, -60f, 60f, 0.35f, Mathf.PI, 0f),
            new("LeftForeArm", "LeftHand", 1.5f, -70f, 10f, 0.15f, Mathf.PI * 1.5f, -0.25f),

            new("RightArm", "RightForeArm", 2.5f, -60f, 60f, 0.35f, 0f, 0f),
            new("RightForeArm", "RightHand", 1.5f, -70f, 10f, 0.15f, Mathf.PI * 0.5f, -0.25f)
        };

        [MenuItem("PoRacer/Build Rigged Creature Prefabs (4 humanoids)")]
        public static void BuildAll()
        {
            int built = 0;
            for (int modelIndex = 0; modelIndex < SourceModels.Length; modelIndex++)
            {
                if (Build(SourceModels[modelIndex]))
                {
                    built++;
                }
            }
            AssetDatabase.SaveAssets();
            Debug.Log($"Built {built} rigged creature prefabs in Assets/Prefabs/.");
        }

        private static bool Build(string modelPath)
        {
            var source = AssetDatabase.LoadAssetAtPath<GameObject>(modelPath);
            if (source == null)
            {
                Debug.LogWarning($"Missing model {modelPath}. Skipping.");
                return false;
            }

            // "RIGGED_Matt.glb" -> behavior "Matt", prefab "Matt_v01".
            string behaviorName = System.IO.Path.GetFileNameWithoutExtension(modelPath).Replace("RIGGED_", string.Empty);

            var model = (GameObject)PrefabUtility.InstantiatePrefab(source);
            PrefabUtility.UnpackPrefabInstance(model, PrefabUnpackMode.Completely, InteractionMode.AutomatedAction);
            model.transform.position = Vector3.zero;
            model.transform.rotation = Quaternion.identity;

            Transform hips = FindBone(model.transform, "Hips");
            if (hips == null)
            {
                Debug.LogWarning($"{modelPath} has no 'Hips' bone. Skipping.");
                Object.DestroyImmediate(model);
                return false;
            }

            // The prefab root sits at hip height and *is* the pelvis body, matching
            // the rest of the fleet (spawn code, camera targets and RacerView all
            // read the prefab root's transform as the creature's position). The
            // imported model hangs underneath it, keeping its 0.01 armature scale.
            var rootGo = new GameObject($"{behaviorName}_v01");
            rootGo.transform.position = hips.position;
            model.transform.SetParent(rootGo.transform, worldPositionStays: true);

            // Skinned bounds are computed from the bind pose; once physics pulls the
            // bones around, an un-updated bound culls the character at odd angles.
            SkinnedMeshRenderer[] skins = rootGo.GetComponentsInChildren<SkinnedMeshRenderer>(true);
            for (int skinIndex = 0; skinIndex < skins.Length; skinIndex++)
            {
                skins[skinIndex].updateWhenOffscreen = true;
            }

            float totalMass = PELVIS_MASS;
            for (int jointIndex = 0; jointIndex < Joints.Length; jointIndex++)
            {
                totalMass += Joints[jointIndex].Mass;
            }

            var pelvis = rootGo.AddComponent<ArticulationBody>();
            pelvis.mass = PELVIS_MASS;
            AddPelvisCollider(rootGo, hips);

            var joints = new List<ArticulationBody>(Joints.Length);
            for (int jointIndex = 0; jointIndex < Joints.Length; jointIndex++)
            {
                JointSpec spec = Joints[jointIndex];
                Transform bone = FindBone(model.transform, spec.Bone);
                if (bone == null)
                {
                    Debug.LogWarning($"{behaviorName}: bone '{spec.Bone}' not found; rig will be incomplete.");
                    continue;
                }
                joints.Add(AttachLimb(bone, FindBone(model.transform, spec.ChildBone), spec, totalMass));
            }

            WireAgent(rootGo, pelvis, joints, behaviorName, totalMass);

            string path = $"Assets/Prefabs/{behaviorName}_v01.prefab";
            PrefabUtility.SaveAsPrefabAsset(rootGo, path);
            Object.DestroyImmediate(rootGo);
            Debug.Log($"Built {path}: {joints.Count} joints, {totalMass:0} kg.");
            return true;
        }

        /// <summary>
        /// Pelvis capsule, sized in world units because the prefab root is not
        /// under the armature's 0.01 scale.
        /// </summary>
        private static void AddPelvisCollider(GameObject rootGo, Transform hips)
        {
            Transform spine = FindBone(hips, "Spine02");
            float height = spine != null ? Vector3.Distance(hips.position, spine.position) * 2f : 0.24f;
            var capsule = rootGo.AddComponent<CapsuleCollider>();
            capsule.direction = 1;
            capsule.height = height;
            capsule.radius = height * 0.42f;
            capsule.center = Vector3.zero;
        }

        /// <summary>
        /// Turns one bone into a driven limb: a capsule spanning bone-to-child and a
        /// revolute joint at the bone's own origin, which in a skeleton is exactly
        /// where the joint belongs — so the anchor is simply zero.
        /// </summary>
        private static ArticulationBody AttachLimb(Transform bone, Transform childBone, JointSpec spec, float totalMass)
        {
            // Bones point down their own local +Y, and lengths stay in armature-local
            // units so the collider scales with the rig like everything else.
            float length = childBone != null ? childBone.localPosition.magnitude : 12f;
            var capsule = bone.gameObject.AddComponent<CapsuleCollider>();
            capsule.direction = 1;
            capsule.height = length;
            capsule.radius = length * LIMB_RADIUS_FRACTION;
            capsule.center = new Vector3(0f, length * 0.5f, 0f);

            var body = bone.gameObject.AddComponent<ArticulationBody>();
            body.mass = spec.Mass;
            body.anchorPosition = Vector3.zero;
            body.anchorRotation = Quaternion.identity;
            body.jointType = ArticulationJointType.RevoluteJoint;
            body.twistLock = ArticulationDofLock.LimitedMotion;

            // Sized by what the joint actually has to move, floored so ground-contact
            // joints are not left driving only their own limb's mass.
            float drivenMass = Mathf.Max(SubtreeMass(spec), totalMass * MIN_DRIVEN_MASS_FRACTION);
            float torque = drivenMass * TORQUE_PER_KG;
            ArticulationDrive drive = body.xDrive;
            drive.lowerLimit = spec.LowerLimit;
            drive.upperLimit = spec.UpperLimit;
            drive.stiffness = torque * STIFFNESS_RATIO;
            drive.damping = torque * DAMPING_RATIO;
            drive.forceLimit = torque;
            drive.target = 0f;
            body.xDrive = drive;
            return body;
        }

        /// <summary>Mass of this joint's limb plus everything hanging off it.</summary>
        private static float SubtreeMass(JointSpec spec)
        {
            float mass = spec.Mass;
            for (int jointIndex = 0; jointIndex < Joints.Length; jointIndex++)
            {
                if (Joints[jointIndex].Bone == spec.ChildBone)
                {
                    mass += SubtreeMass(Joints[jointIndex]);
                }
            }
            return mass;
        }

        private static void WireAgent(GameObject rootGo, ArticulationBody pelvis, List<ArticulationBody> joints,
            string behaviorName, float totalMass)
        {
            var behavior = rootGo.AddComponent<BehaviorParameters>();
            behavior.BehaviorName = behaviorName;
            behavior.BrainParameters.VectorObservationSize = joints.Count * 3 + 19;
            behavior.BrainParameters.NumStackedVectorObservations = 1;
            behavior.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(joints.Count);

            var agent = rootGo.AddComponent<Agent_Creature>();
            var so = new SerializedObject(agent);
            so.FindProperty("_root").objectReferenceValue = pelvis;
            SerializedProperty jointList = so.FindProperty("_joints");
            jointList.arraySize = joints.Count;
            for (int jointIndex = 0; jointIndex < joints.Count; jointIndex++)
            {
                jointList.GetArrayElementAtIndex(jointIndex).objectReferenceValue = joints[jointIndex];
            }
            so.FindProperty("_jointDriveScale").floatValue = JOINT_DRIVE_SCALE;
            so.FindProperty("_maxJointTorque").floatValue = totalMass * TORQUE_PER_KG;
            so.FindProperty("_gaitFrequency").floatValue = 1.1f;

            SerializedProperty phases = so.FindProperty("_gaitPhases");
            SerializedProperty amplitudes = so.FindProperty("_gaitAmplitudes");
            SerializedProperty offsets = so.FindProperty("_gaitOffsets");
            phases.arraySize = joints.Count;
            amplitudes.arraySize = joints.Count;
            offsets.arraySize = joints.Count;
            for (int jointIndex = 0; jointIndex < joints.Count; jointIndex++)
            {
                JointSpec spec = Joints[jointIndex];
                phases.GetArrayElementAtIndex(jointIndex).floatValue = spec.GaitPhase;
                amplitudes.GetArrayElementAtIndex(jointIndex).floatValue = spec.GaitAmplitude;
                offsets.GetArrayElementAtIndex(jointIndex).floatValue = spec.GaitOffset;
            }
            so.ApplyModifiedPropertiesWithoutUndo();

            var requester = rootGo.AddComponent<DecisionRequester>();
            requester.DecisionPeriod = DECISION_PERIOD;
            requester.TakeActionsBetweenDecisions = true;
        }

        private static Transform FindBone(Transform root, string boneName)
        {
            Transform[] all = root.GetComponentsInChildren<Transform>(true);
            for (int transformIndex = 0; transformIndex < all.Length; transformIndex++)
            {
                if (all[transformIndex].name == boneName)
                {
                    return all[transformIndex];
                }
            }
            return null;
        }
    }
}
#endif
