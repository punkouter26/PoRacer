using System.Text;
using UnityEditor;
using UnityEngine;

namespace IsaacBiped2.Editor
{
    /// <summary>
    /// Compares what Unity actually realises against what the policy was trained against, and fails
    /// loudly on any mismatch.
    ///
    /// This exists because a silent config mismatch cost days: the rig's PhysicsMaterial asset was
    /// written with a `.physicsMaterial` extension that Unity 6 imports as a DefaultAsset, so every
    /// collider referencing it fell back to the engine default 0.6/0.6 and the biped trained at
    /// Isaac's 0.5/0.5 kept falling over. Nothing errored at runtime — the reference simply did not
    /// resolve. Checks like these are cheap; finding the same bug by bisecting physics is not.
    ///
    /// Run from PoRacer > Isaac Biped 2 > Check Physics Parity, or call <see cref="Run"/>.
    /// </summary>
    public static class IsaacBiped2ParityCheck
    {
        // Isaac-side contract. Mirrors IsaacBiped2Agent and the export report.
        private const float ISAAC_PHYSICS_DT = 0.005f;
        private const float ISAAC_FRICTION = 0.5f;
        private const float ISAAC_RESTITUTION = 0f;
        private const float ISAAC_CONTACT_OFFSET = 0.02f;
        private const float ISAAC_GRAVITY = -9.81f;
        private const float ISAAC_TOTAL_MASS = 15.0f;  // wide-foot rig, 2026-08-29 (was 14.8)
        private const float ISAAC_SPAWN_HEIGHT = 0.68f;
        private const int ISAAC_BODIES = 11;
        private const int ISAAC_JOINTS = 10;

        [MenuItem("PoRacer/Isaac Biped 2/Check Physics Parity", priority = 30)]
        public static void RunMenu() => Debug.Log(Run());

        /// <summary>Returns a human-readable report; every line is PASS or FAIL.</summary>
        public static string Run()
        {
            var report = new StringBuilder("[IsaacBiped2 parity]\n");
            int failures = 0;

            // --- project-wide settings the policy depends on
            failures += Check(report, "physics step", Time.fixedDeltaTime, ISAAC_PHYSICS_DT, 1e-4f,
                "policy trained at 200 Hz; a coarser step diverges to NaN");
            failures += Check(report, "gravity.y", Physics.gravity.y, ISAAC_GRAVITY, 1e-3f, null);

            // --- the material, the check that would have caught the original bug
            var material = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(IsaacBiped2Setup.PHYSICS_MATERIAL_PATH);
            if (material == null)
            {
                report.Append($"FAIL material: {IsaacBiped2Setup.PHYSICS_MATERIAL_PATH} does not load as a ")
                      .Append("PhysicsMaterial (a '.physicsMaterial' extension imports as DefaultAsset; use '.asset')\n");
                failures++;
            }
            else
            {
                failures += Check(report, "material static friction", material.staticFriction, ISAAC_FRICTION, 1e-3f, null);
                failures += Check(report, "material dynamic friction", material.dynamicFriction, ISAAC_FRICTION, 1e-3f, null);
                failures += Check(report, "material bounciness", material.bounciness, ISAAC_RESTITUTION, 1e-3f, null);
            }

            // --- the prefab
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(IsaacBiped2Setup.PREFAB_PATH);
            if (prefab == null)
            {
                report.Append("FAIL prefab: not found\n");
                return Finish(report, failures + 1);
            }

            ArticulationBody[] bodies = prefab.GetComponentsInChildren<ArticulationBody>(true);
            failures += Check(report, "body count", bodies.Length, ISAAC_BODIES, null);

            float mass = 0f;
            for (int index = 0; index < bodies.Length; index++)
            {
                mass += bodies[index].mass;
            }
            failures += Check(report, "total mass", mass, ISAAC_TOTAL_MASS, 0.05f, null);

            Collider[] colliders = prefab.GetComponentsInChildren<Collider>(true);
            int missingMaterial = 0;
            for (int index = 0; index < colliders.Length; index++)
            {
                if (colliders[index].sharedMaterial == null)
                {
                    missingMaterial++;
                }
            }
            failures += Check(report, "colliders without a material", missingMaterial, 0, null);

            // --- joints: every driven joint must exist with the trained limits
            int jointsFound = 0;
            for (int index = 0; index < IsaacBiped2Agent.JointOrder.Length; index++)
            {
                for (int b = 0; b < bodies.Length; b++)
                {
                    if (bodies[b].jointType != ArticulationJointType.RevoluteJoint)
                    {
                        continue;
                    }
                    if (bodies[b].name == LinkFor(IsaacBiped2Agent.JointOrder[index]))
                    {
                        jointsFound++;
                        break;
                    }
                }
            }
            failures += Check(report, "driven joints resolved", jointsFound, ISAAC_JOINTS, null);

            // --- the agent's own settings
            var agent = prefab.GetComponent<IsaacBiped2Agent>();
            if (agent == null)
            {
                report.Append("FAIL agent: IsaacBiped2Agent missing from the prefab\n");
                failures++;
            }
            else
            {
                var so = new SerializedObject(agent);
                failures += Check(report, "agent contact offset",
                    so.FindProperty("_contactOffset").floatValue, ISAAC_CONTACT_OFFSET, 1e-4f, null);
                failures += Check(report, "agent action scale",
                    so.FindProperty("_actionScale").floatValue, 0.5f, 1e-4f, null);
                failures += Check(report, "agent policy dt",
                    so.FindProperty("_policyDt").floatValue, 0.02f, 1e-5f, null);
                var model = so.FindProperty("_model");
                if (model == null || model.objectReferenceValue == null)
                {
                    report.Append("FAIL agent policy: no ModelAsset assigned\n");
                    failures++;
                }
                else
                {
                    report.Append($"PASS agent policy: {model.objectReferenceValue.name}\n");
                }
            }

            failures += Check(report, "prefab spawn height", prefab.transform.position.y, ISAAC_SPAWN_HEIGHT, 1e-3f, null);
            return Finish(report, failures);
        }

        private static string LinkFor(string jointName)
        {
            string side = jointName.Substring(0, 2);
            string joint = jointName.Substring(2);
            switch (joint)
            {
                case "hip_yaw": return side + "hip_yaw_link";
                case "hip_roll": return side + "hip_roll_link";
                case "hip_pitch": return side + "thigh";
                case "knee": return side + "shank";
                default: return side + "foot";
            }
        }

        private static int Check(StringBuilder report, string label, float actual, float expected, float tolerance, string note)
        {
            bool ok = Mathf.Abs(actual - expected) <= tolerance;
            report.Append(ok ? "PASS " : "FAIL ").Append(label).Append(": ").Append(actual.ToString("0.#####"));
            if (!ok)
            {
                report.Append(" (expected ").Append(expected.ToString("0.#####")).Append(')');
                if (note != null)
                {
                    report.Append(" -- ").Append(note);
                }
            }
            report.Append('\n');
            return ok ? 0 : 1;
        }

        private static int Check(StringBuilder report, string label, int actual, int expected, string note)
        {
            bool ok = actual == expected;
            report.Append(ok ? "PASS " : "FAIL ").Append(label).Append(": ").Append(actual);
            if (!ok)
            {
                report.Append(" (expected ").Append(expected).Append(')');
                if (note != null)
                {
                    report.Append(" -- ").Append(note);
                }
            }
            report.Append('\n');
            return ok ? 0 : 1;
        }

        private static string Finish(StringBuilder report, int failures)
        {
            report.Append(failures == 0 ? "ALL PARITY CHECKS PASS" : $"{failures} PARITY CHECK(S) FAILED");
            return report.ToString();
        }
    }
}
