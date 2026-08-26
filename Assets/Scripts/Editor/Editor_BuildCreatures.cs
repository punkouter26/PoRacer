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
    /// Procedurally builds the 7 new creature prefabs (hexapod, quad, snake,
    /// centipede, crab, kangaroo, blob): primitive bodies, ArticulationBody chains
    /// with anchors derived from segment lengths, drives sized to total mass with
    /// the same stiffness/damping/force ratios as the tuned worm prefab, and
    /// ML-Agents wiring (BehaviorParameters + DecisionRequester + Agent_Creature).
    /// Per-joint force limit follows the fleet rule: totalMass x 10 N*m.
    /// </summary>
    public static class Editor_BuildCreatures
    {
        private const int DECISION_PERIOD = 5;
        private const float TORQUE_PER_KG = 10f;

        private sealed class BuildContext
        {
            public readonly List<ArticulationBody> Joints = new();
            public float Torque;
            public float StiffnessRatio;
            public float DampingRatio;
        }

        [MenuItem("PoRacer/Build Creature Prefabs (7 new)")]
        public static void BuildAll()
        {
            (float stiffnessRatio, float dampingRatio) = ReadWormDriveRatios();
            BuildHexapod("Hexapod", hipAxisY: false, stiffnessRatio, dampingRatio);
            BuildQuad(stiffnessRatio, dampingRatio);
            BuildSnake("Snake", segmentCount: 10, segmentScale: new Vector3(0.14f, 0.22f, 0.14f), segmentMass: 6f, stiffnessRatio, dampingRatio);
            BuildCentipede(stiffnessRatio, dampingRatio);
            BuildHexapod("Crab", hipAxisY: true, stiffnessRatio, dampingRatio);
            BuildKangaroo(stiffnessRatio, dampingRatio);
            BuildBlob(stiffnessRatio, dampingRatio);
            AssetDatabase.SaveAssets();
            Debug.Log("Built 7 creature prefabs in Assets/Prefabs/.");
        }

        private static (float, float) ReadWormDriveRatios()
        {
            var worm = AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Prefabs/Worm_v01.prefab");
            foreach (ArticulationBody body in worm.GetComponentsInChildren<ArticulationBody>(true))
            {
                if (body.jointType == ArticulationJointType.RevoluteJoint && body.xDrive.forceLimit > 0f && !float.IsInfinity(body.xDrive.forceLimit))
                {
                    return (body.xDrive.stiffness / body.xDrive.forceLimit, body.xDrive.damping / body.xDrive.forceLimit);
                }
            }
            Debug.LogWarning("No tuned worm drive found; using fallback ratios.");
            return (20f, 1f);
        }

        // --- shared part helpers -------------------------------------------------

        private static GameObject Part(PrimitiveType type, string name, Transform parent, Vector3 position, Vector3 euler, Vector3 scale)
        {
            var go = GameObject.CreatePrimitive(type);
            go.name = name;
            go.transform.SetParent(parent, false);
            go.transform.localPosition = position;
            go.transform.localEulerAngles = euler;
            go.transform.localScale = scale;
            return go;
        }

        private static ArticulationBody MakeRoot(GameObject go, float mass)
        {
            var body = go.AddComponent<ArticulationBody>();
            body.mass = mass;
            return body;
        }

        private static ArticulationBody Hinge(BuildContext context, GameObject go, float mass, Vector3 anchorLocal, Quaternion anchorRotation, float limitDegrees)
        {
            ArticulationBody body = AttachJoint(context, go, mass, anchorLocal, anchorRotation);
            body.jointType = ArticulationJointType.RevoluteJoint;
            body.twistLock = ArticulationDofLock.LimitedMotion;
            ArticulationDrive drive = body.xDrive;
            drive.lowerLimit = -limitDegrees;
            drive.upperLimit = limitDegrees;
            drive.stiffness = context.Torque * context.StiffnessRatio;
            drive.damping = context.Torque * context.DampingRatio;
            drive.forceLimit = context.Torque;
            drive.target = 0f;
            body.xDrive = drive;
            return body;
        }

        private static ArticulationBody Slider(BuildContext context, GameObject go, float mass, Vector3 anchorLocal, Quaternion anchorRotation, float limitMetres)
        {
            ArticulationBody body = AttachJoint(context, go, mass, anchorLocal, anchorRotation);
            body.jointType = ArticulationJointType.PrismaticJoint;
            body.linearLockX = ArticulationDofLock.LimitedMotion;
            body.linearLockY = ArticulationDofLock.LockedMotion;
            body.linearLockZ = ArticulationDofLock.LockedMotion;
            ArticulationDrive drive = body.xDrive;
            drive.lowerLimit = -limitMetres;
            drive.upperLimit = limitMetres;
            drive.stiffness = context.Torque * context.StiffnessRatio;
            drive.damping = context.Torque * context.DampingRatio;
            drive.forceLimit = context.Torque;
            drive.target = 0f;
            body.xDrive = drive;
            return body;
        }

        private static ArticulationBody AttachJoint(BuildContext context, GameObject go, float mass, Vector3 anchorLocal, Quaternion anchorRotation)
        {
            var body = go.AddComponent<ArticulationBody>();
            body.mass = mass;
            body.anchorPosition = anchorLocal;
            body.anchorRotation = anchorRotation;
            context.Joints.Add(body);
            return body;
        }

        private static Quaternion AxisX => Quaternion.identity;
        private static Quaternion AxisY => Quaternion.Euler(0f, 0f, 90f);
        private static Quaternion AxisZ => Quaternion.Euler(0f, -90f, 0f);

        private static void Finish(string behaviorName, GameObject rootGo, BuildContext context, float driveScale)
        {
            Finish(behaviorName, rootGo, context, driveScale, 0.8f, null, null);
        }

        private static void Finish(string behaviorName, GameObject rootGo, BuildContext context, float driveScale,
            float gaitFrequency, float[] gaitPhases, float[] gaitAmplitudes)
        {
            var behavior = rootGo.AddComponent<BehaviorParameters>();
            behavior.BehaviorName = behaviorName;
            // Must match Agent_Creature's layout: per joint position + velocity +
            // contact, plus 19 root/goal/stamina/probe values. Editor_SyncObservationSizes
            // rewrites existing prefabs after a layout change.
            behavior.BrainParameters.VectorObservationSize = context.Joints.Count * 3 + 19;
            behavior.BrainParameters.NumStackedVectorObservations = 1;
            behavior.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(context.Joints.Count);

            var agent = rootGo.AddComponent<Agent_Creature>();
            var so = new SerializedObject(agent);
            so.FindProperty("_root").objectReferenceValue = rootGo.GetComponent<ArticulationBody>();
            SerializedProperty joints = so.FindProperty("_joints");
            joints.arraySize = context.Joints.Count;
            for (int jointIndex = 0; jointIndex < context.Joints.Count; jointIndex++)
            {
                joints.GetArrayElementAtIndex(jointIndex).objectReferenceValue = context.Joints[jointIndex];
            }
            so.FindProperty("_jointDriveScale").floatValue = driveScale;
            so.FindProperty("_maxJointTorque").floatValue = context.Torque;
            so.FindProperty("_gaitFrequency").floatValue = gaitFrequency;
            SerializedProperty phases = so.FindProperty("_gaitPhases");
            SerializedProperty amplitudes = so.FindProperty("_gaitAmplitudes");
            int gaitLength = gaitPhases != null ? gaitPhases.Length : 0;
            phases.arraySize = gaitLength;
            amplitudes.arraySize = gaitLength;
            for (int gaitIndex = 0; gaitIndex < gaitLength; gaitIndex++)
            {
                phases.GetArrayElementAtIndex(gaitIndex).floatValue = gaitPhases[gaitIndex];
                amplitudes.GetArrayElementAtIndex(gaitIndex).floatValue = gaitAmplitudes[gaitIndex];
            }
            so.ApplyModifiedPropertiesWithoutUndo();

            var requester = rootGo.AddComponent<DecisionRequester>();
            requester.DecisionPeriod = DECISION_PERIOD;
            requester.TakeActionsBetweenDecisions = true;

            string path = $"Assets/Prefabs/{behaviorName}_v01.prefab";
            PrefabUtility.SaveAsPrefabAsset(rootGo, path);
            Object.DestroyImmediate(rootGo);
            Debug.Log($"Built {path}: {context.Joints.Count} joints, torque {context.Torque:0} N*m.");
        }

        // --- creatures -----------------------------------------------------------

        private static void BuildHexapod(string behaviorName, bool hipAxisY, float stiffnessRatio, float dampingRatio)
        {
            var context = new BuildContext { Torque = 70f * TORQUE_PER_KG, StiffnessRatio = stiffnessRatio, DampingRatio = dampingRatio };
            var rootGo = Part(PrimitiveType.Cube, behaviorName, null, new Vector3(0f, 0.7f, 0f), Vector3.zero, new Vector3(1.1f, 0.24f, 0.7f));
            MakeRoot(rootGo, 40f);
            Quaternion hipAxis = hipAxisY ? AxisY : AxisX;
            var phases = new List<float>();
            var amplitudes = new List<float>();
            for (int side = -1; side <= 1; side += 2)
            {
                for (int legIndex = 0; legIndex < 3; legIndex++)
                {
                    float legZ = (legIndex - 1) * 0.28f;
                    var upper = Part(PrimitiveType.Capsule, $"Upper_{side}_{legIndex}", rootGo.transform,
                        new Vector3(side * 0.62f, -0.3f, legZ / 0.24f), Vector3.zero, new Vector3(0.33f, 0.63f, 0.43f));
                    Hinge(context, upper, 3f, new Vector3(0f, 1f, 0f), hipAxis, 40f);
                    var lower = Part(PrimitiveType.Capsule, $"Lower_{side}_{legIndex}", upper.transform,
                        new Vector3(0f, -2f, 0f), Vector3.zero, new Vector3(0.8f, 1f, 0.8f));
                    Hinge(context, lower, 2f, new Vector3(0f, 1f, 0f), AxisX, 50f);
                    // Tripod gait: alternate legs form two groups half a cycle apart;
                    // the knee leads the hip by a quarter cycle to lift before the swing.
                    float groupPhase = (side + 1) / 2 + legIndex % 2 == 1 ? 0f : Mathf.PI;
                    phases.Add(groupPhase);
                    amplitudes.Add(0.7f);
                    phases.Add(groupPhase + Mathf.PI * 0.5f);
                    amplitudes.Add(0.5f);
                }
            }
            Finish(behaviorName, rootGo, context, 45f, 0.9f, phases.ToArray(), amplitudes.ToArray());
        }

        private static void BuildQuad(float stiffnessRatio, float dampingRatio)
        {
            var context = new BuildContext { Torque = 90f * TORQUE_PER_KG, StiffnessRatio = stiffnessRatio, DampingRatio = dampingRatio };
            var rootGo = Part(PrimitiveType.Cube, "Quad", null, new Vector3(0f, 0.8f, 0f), Vector3.zero, new Vector3(0.9f, 0.3f, 1.1f));
            MakeRoot(rootGo, 60f);
            var phases = new List<float>();
            var amplitudes = new List<float>();
            for (int side = -1; side <= 1; side += 2)
            {
                for (int end = -1; end <= 1; end += 2)
                {
                    var upper = Part(PrimitiveType.Capsule, $"Upper_{side}_{end}", rootGo.transform,
                        new Vector3(side * 0.55f, -1.2f, end * 0.41f), Vector3.zero, new Vector3(0.2f, 0.6f, 0.16f));
                    Hinge(context, upper, 4f, new Vector3(0f, 1f, 0f), AxisX, 45f);
                    var lower = Part(PrimitiveType.Capsule, $"Lower_{side}_{end}", upper.transform,
                        new Vector3(0f, -2f, 0f), Vector3.zero, new Vector3(0.8f, 1f, 0.8f));
                    Hinge(context, lower, 3.5f, new Vector3(0f, 1f, 0f), AxisX, 60f);
                    // Trot: diagonal leg pairs move together, half a cycle apart.
                    float groupPhase = side * end > 0 ? 0f : Mathf.PI;
                    phases.Add(groupPhase);
                    amplitudes.Add(0.7f);
                    phases.Add(groupPhase + Mathf.PI * 0.5f);
                    amplitudes.Add(0.5f);
                }
            }
            Finish("Quad", rootGo, context, 45f, 1.0f, phases.ToArray(), amplitudes.ToArray());
        }

        private static void BuildSnake(string behaviorName, int segmentCount, Vector3 segmentScale, float segmentMass, float stiffnessRatio, float dampingRatio)
        {
            var context = new BuildContext { Torque = segmentCount * segmentMass * TORQUE_PER_KG, StiffnessRatio = stiffnessRatio, DampingRatio = dampingRatio };
            var rootGo = Part(PrimitiveType.Capsule, behaviorName, null, new Vector3(0f, segmentScale.x, 0f), new Vector3(90f, 0f, 0f), segmentScale);
            MakeRoot(rootGo, segmentMass);
            Transform previous = rootGo.transform;
            var phases = new List<float>();
            var amplitudes = new List<float>();
            for (int segmentIndex = 1; segmentIndex < segmentCount; segmentIndex++)
            {
                bool yaw = segmentIndex % 2 == 0;
                var segment = Part(PrimitiveType.Capsule, $"Segment_{segmentIndex}", previous,
                    new Vector3(0f, -2f, 0f), Vector3.zero, Vector3.one);
                // Alternate pitch (climb hills) and yaw (steer around obstacles).
                Hinge(context, segment, segmentMass, new Vector3(0f, 1f, 0f), yaw ? AxisY : AxisX, 40f);
                previous = segment.transform;
                // Serpentine: a wave travels head to tail, which is what drives an
                // undulating body forward. The phase must therefore *lead* with
                // depth — a negative step runs the wave tail to head and walks the
                // creature backwards (measured: worm -0.3 m vs +2.1 m per 30 s).
                phases.Add(segmentIndex * 0.7f);
                amplitudes.Add(yaw ? 0.85f : 0.3f);
            }
            Finish(behaviorName, rootGo, context, 40f, 0.7f, phases.ToArray(), amplitudes.ToArray());
        }

        private static void BuildCentipede(float stiffnessRatio, float dampingRatio)
        {
            var context = new BuildContext { Torque = 80f * TORQUE_PER_KG, StiffnessRatio = stiffnessRatio, DampingRatio = dampingRatio };
            var segmentScale = new Vector3(0.16f, 0.25f, 0.16f);
            var rootGo = Part(PrimitiveType.Capsule, "Centipede", null, new Vector3(0f, 0.2f, 0f), new Vector3(90f, 0f, 0f), segmentScale);
            MakeRoot(rootGo, 8f);
            Transform previous = rootGo.transform;
            var segments = new List<Transform> { rootGo.transform };
            var phases = new List<float>();
            var amplitudes = new List<float>();
            for (int segmentIndex = 1; segmentIndex < 6; segmentIndex++)
            {
                var segment = Part(PrimitiveType.Capsule, $"Segment_{segmentIndex}", previous,
                    new Vector3(0f, -2f, 0f), Vector3.zero, Vector3.one);
                Hinge(context, segment, 8f, new Vector3(0f, 1f, 0f), segmentIndex % 2 == 0 ? AxisY : AxisX, 30f);
                segments.Add(segment.transform);
                previous = segment.transform;
                // Gentle spine wave so the body follows the legs; head-to-tail for
                // the same reason as the snake's.
                phases.Add(segmentIndex * 0.6f);
                amplitudes.Add(0.25f);
            }
            for (int segmentIndex = 0; segmentIndex < segments.Count; segmentIndex++)
            {
                for (int side = -1; side <= 1; side += 2)
                {
                    var leg = Part(PrimitiveType.Capsule, $"Leg_{side}", segments[segmentIndex],
                        new Vector3(side * 1.4f, 0f, 0f), new Vector3(0f, 0f, side * 75f), new Vector3(0.35f, 0.55f, 0.35f));
                    // Legs paddle around Y: sweep back to push, lift on the return.
                    Hinge(context, leg, 1.5f, new Vector3(0f, 1f, 0f), AxisY, 35f);
                    // Metachronal wave: each segment's legs lag the pair ahead of it,
                    // left and right half a cycle apart.
                    phases.Add(segmentIndex * 0.9f + (side > 0 ? Mathf.PI : 0f));
                    amplitudes.Add(0.8f);
                }
            }
            Finish("Centipede", rootGo, context, 35f, 0.9f, phases.ToArray(), amplitudes.ToArray());
        }

        private static void BuildKangaroo(float stiffnessRatio, float dampingRatio)
        {
            var context = new BuildContext { Torque = 85f * TORQUE_PER_KG, StiffnessRatio = stiffnessRatio, DampingRatio = dampingRatio };
            var rootGo = Part(PrimitiveType.Capsule, "Kangaroo", null, new Vector3(0f, 1.0f, 0f), Vector3.zero, new Vector3(0.5f, 0.45f, 0.5f));
            MakeRoot(rootGo, 45f);
            for (int side = -1; side <= 1; side += 2)
            {
                var thigh = Part(PrimitiveType.Capsule, $"Thigh_{side}", rootGo.transform,
                    new Vector3(side * 0.34f, -1f, 0f), Vector3.zero, new Vector3(0.4f, 0.44f, 0.4f));
                Hinge(context, thigh, 8f, new Vector3(0f, 1f, 0f), AxisX, 70f);
                var shin = Part(PrimitiveType.Capsule, $"Shin_{side}", thigh.transform,
                    new Vector3(0f, -2f, 0f), Vector3.zero, new Vector3(0.8f, 1f, 0.8f));
                Hinge(context, shin, 5f, new Vector3(0f, 1f, 0f), AxisX, 80f);
            }
            var tailBase = Part(PrimitiveType.Capsule, "TailBase", rootGo.transform,
                new Vector3(0f, -0.7f, -0.9f), new Vector3(-65f, 0f, 0f), new Vector3(0.4f, 0.44f, 0.4f));
            Hinge(context, tailBase, 6f, new Vector3(0f, 1f, 0f), AxisX, 40f);
            var tailTip = Part(PrimitiveType.Capsule, "TailTip", tailBase.transform,
                new Vector3(0f, -2f, 0f), Vector3.zero, new Vector3(0.75f, 0.9f, 0.75f));
            Hinge(context, tailTip, 4f, new Vector3(0f, 1f, 0f), AxisX, 40f);
            // Hop cycle: both legs together (thighs lead, shins follow a quarter
            // cycle later), tail swings counter-phase as the balance arm.
            float[] phases = { 0f, Mathf.PI * 0.5f, 0f, Mathf.PI * 0.5f, Mathf.PI, Mathf.PI + 0.5f };
            float[] amplitudes = { 0.8f, 0.7f, 0.8f, 0.7f, 0.5f, 0.4f };
            Finish("Kangaroo", rootGo, context, 60f, 1.2f, phases, amplitudes);
        }

        private static void BuildBlob(float stiffnessRatio, float dampingRatio)
        {
            var context = new BuildContext { Torque = 50f * TORQUE_PER_KG, StiffnessRatio = stiffnessRatio, DampingRatio = dampingRatio };
            var rootGo = Part(PrimitiveType.Sphere, "Blob", null, new Vector3(0f, 0.35f, 0f), Vector3.zero, new Vector3(0.6f, 0.6f, 0.6f));
            MakeRoot(rootGo, 30f);
            for (int armIndex = 0; armIndex < 5; armIndex++)
            {
                float angle = armIndex * 72f;
                Quaternion outward = Quaternion.Euler(0f, angle, 0f);
                Vector3 direction = outward * Vector3.forward;
                var arm = Part(PrimitiveType.Sphere, $"Arm_{armIndex}", rootGo.transform,
                    direction * 1.2f, new Vector3(0f, angle, 0f), new Vector3(0.5f, 0.5f, 0.5f));
                // Prismatic along the arm's outward axis: the blob squeezes to crawl.
                Slider(context, arm, 4f, Vector3.zero, AxisZ, 0.35f);
            }
            // Rotating squeeze: arms extend one after another around the circle.
            var phases = new float[5];
            var amplitudes = new float[5];
            for (int armIndex = 0; armIndex < 5; armIndex++)
            {
                phases[armIndex] = -armIndex * (2f * Mathf.PI / 5f);
                amplitudes[armIndex] = 1f;
            }
            Finish("Blob", rootGo, context, 0.3f, 0.6f, phases, amplitudes);
        }
    }
}
#endif
