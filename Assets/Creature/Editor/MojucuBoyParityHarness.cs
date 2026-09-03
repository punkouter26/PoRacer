using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Creature.MojucuBoy;
using Mujoco;
using UnityEngine;

namespace CreatureEditor
{
    /// <summary>
    /// Phase 3 parity harness: load the round-trip MJCF in Unity, force it into the
    /// exact state the Python side used, and write the resulting observation vector
    /// out for an element-wise comparison.
    ///
    /// The state is supplied as raw qpos/qvel rather than as Unity transforms on
    /// purpose. qpos/qvel is what MuJoCo integrates and what the policy will be
    /// trained against, so it is the only representation where "identical state" is
    /// unambiguous; going through Unity transforms would fold MjEngineTool's frame
    /// conversion into the test and make a failure ambiguous between the state
    /// transfer and the observation itself.
    ///
    /// Called from the CLI:
    ///   unity cmd eval --code "CreatureEditor.MojucuBoyParityHarness.Run();"
    /// because the eval host compiles without /unsafe and cannot touch mjData.
    /// </summary>
    public static class MojucuBoyParityHarness
    {
        private const string STATE_FILE = "training/mojucuboy/_parity_state.json";
        private const string OUT_FILE = "training/mojucuboy/_parity_unity.json";

        public static string Run()
        {
            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            string statePath = Path.Combine(projectRoot, STATE_FILE.Replace('/', Path.DirectorySeparatorChar));
            string outPath = Path.Combine(projectRoot, OUT_FILE.Replace('/', Path.DirectorySeparatorChar));

            if (!File.Exists(statePath))
            {
                return $"ERROR: no state file at {statePath} -- run parity_state.py first";
            }
            State state = State.Parse(File.ReadAllText(statePath));

            if (!MjScene.InstanceExists)
            {
                return "ERROR: no MjScene in the open scene -- open SCN_MOJUCUBOY_RIGTEST first";
            }
            MjScene scene = MjScene.Instance;

            // Rebuild so the model is compiled and mjData allocated. skipCompile:false
            // is the whole point -- we want Unity's own compiled model, not a fresh
            // Python one.
            scene.CreateScene(false);

            return Evaluate(scene, state, outPath);
        }

        private static unsafe string Evaluate(MjScene scene, State state, string outPath)
        {
            MujocoLib.mjModel_* model = scene.Model;
            MujocoLib.mjData_* data = scene.Data;
            if (model == null || data == null)
            {
                return "ERROR: MjScene has no compiled model/data";
            }

            // The generated bindings type these as ulong.
            int nq = (int)model->nq;
            int nv = (int)model->nv;
            int nu = (int)model->nu;
            if (state.Qpos.Length != nq || state.Qvel.Length != nv)
            {
                return $"ERROR: state is nq={state.Qpos.Length} nv={state.Qvel.Length}, "
                     + $"model is nq={nq} nv={nv}";
            }

            for (int i = 0; i < nq; i++) { data->qpos[i] = state.Qpos[i]; }
            for (int i = 0; i < nv; i++) { data->qvel[i] = state.Qvel[i]; }
            for (int i = 0; i < nu; i++) { data->ctrl[i] = 0.0; }
            MujocoLib.mj_forward(model, data);

            // Resolve the root body and the per-actuator qpos/dof addresses by NAME,
            // never by index: org.mujoco renames every element on export, and the
            // actuator order is the contract, not the declaration order.
            int rootBodyId = BodyId(model, state.RootBody);
            if (rootBodyId < 0)
            {
                return $"ERROR: root body '{state.RootBody}' not found in the Unity model";
            }

            var qposAddr = new int[state.ActuatorOrder.Length];
            var dofAddr = new int[state.ActuatorOrder.Length];
            var missing = new List<string>();
            for (int i = 0; i < state.ActuatorOrder.Length; i++)
            {
                int jointId = JointId(model, state.ActuatorOrder[i]);
                if (jointId < 0) { missing.Add(state.ActuatorOrder[i]); continue; }
                qposAddr[i] = model->jnt_qposadr[jointId];
                dofAddr[i] = model->jnt_dofadr[jointId];
            }
            if (missing.Count > 0)
            {
                return "ERROR: joints not found in the Unity model: " + string.Join(",", missing);
            }

            var obs = new float[MojucuBoyObservation.OBS_SIZE];
            MojucuBoyObservation.Build(data, rootBodyId, qposAddr, dofAddr,
                                 state.CommandHeading, state.CommandSpeed, state.LastAction, obs);

            var sb = new StringBuilder();
            sb.Append("{\n  \"nq\": ").Append(nq)
              .Append(",\n  \"nv\": ").Append(nv)
              .Append(",\n  \"nu\": ").Append(nu)
              .Append(",\n  \"root_body_id\": ").Append(rootBodyId)
              .Append(",\n  \"timestep\": ").Append(Fmt(model->opt.timestep))
              .Append(",\n  \"obs\": [");
            for (int i = 0; i < obs.Length; i++)
            {
                if (i > 0) { sb.Append(", "); }
                sb.Append(Fmt(obs[i]));
            }
            sb.Append("]\n}\n");
            File.WriteAllText(outPath, sb.ToString());

            return $"wrote {outPath} (nq={nq} nv={nv} nu={nu} rootBodyId={rootBodyId})";
        }

        private static string Fmt(double value)
        {
            return value.ToString("R", CultureInfo.InvariantCulture);
        }

        private static unsafe int BodyId(MujocoLib.mjModel_* model, string name)
        {
            return MujocoLib.mj_name2id(model, (int)MujocoLib.mjtObj.mjOBJ_BODY, name);
        }

        private static unsafe int JointId(MujocoLib.mjModel_* model, string name)
        {
            return MujocoLib.mj_name2id(model, (int)MujocoLib.mjtObj.mjOBJ_JOINT, name);
        }

        /// <summary>
        /// Minimal reader for the flat JSON the Python side writes. Deliberately not
        /// JsonUtility: that cannot deserialise a bare double[] field reliably across
        /// versions, and a silent zero-filled array here would look like a parity
        /// failure in the physics.
        /// </summary>
        private sealed class State
        {
            public double[] Qpos;
            public double[] Qvel;
            public string[] ActuatorOrder;
            public string RootBody;
            public float CommandHeading;
            public float CommandSpeed;
            public float[] LastAction;

            public static State Parse(string json)
            {
                return new State
                {
                    Qpos = Doubles(json, "qpos"),
                    Qvel = Doubles(json, "qvel"),
                    ActuatorOrder = Strings(json, "actuator_order"),
                    RootBody = Scalar(json, "root_body").Trim('"'),
                    CommandHeading = (float)double.Parse(Scalar(json, "command_heading"),
                                                         CultureInfo.InvariantCulture),
                    CommandSpeed = (float)double.Parse(Scalar(json, "command_speed"),
                                                       CultureInfo.InvariantCulture),
                    LastAction = Floats(json, "last_action"),
                };
            }

            private static string Body(string json, string key)
            {
                int at = json.IndexOf("\"" + key + "\"", StringComparison.Ordinal);
                if (at < 0) { throw new FormatException($"key '{key}' missing"); }
                int colon = json.IndexOf(':', at);
                return json.Substring(colon + 1);
            }

            private static string Scalar(string json, string key)
            {
                string rest = Body(json, key);
                int end = rest.IndexOfAny(new[] { ',', '}', '\n' });
                return rest.Substring(0, end).Trim();
            }

            private static string[] Elements(string json, string key)
            {
                string rest = Body(json, key);
                int open = rest.IndexOf('[');
                int close = rest.IndexOf(']', open);
                string inner = rest.Substring(open + 1, close - open - 1).Trim();
                if (inner.Length == 0) { return Array.Empty<string>(); }
                string[] parts = inner.Split(',');
                for (int i = 0; i < parts.Length; i++) { parts[i] = parts[i].Trim(); }
                return parts;
            }

            private static double[] Doubles(string json, string key)
            {
                string[] parts = Elements(json, key);
                var values = new double[parts.Length];
                for (int i = 0; i < parts.Length; i++)
                {
                    values[i] = double.Parse(parts[i], CultureInfo.InvariantCulture);
                }
                return values;
            }

            private static float[] Floats(string json, string key)
            {
                double[] values = Doubles(json, key);
                var floats = new float[values.Length];
                for (int i = 0; i < values.Length; i++) { floats[i] = (float)values[i]; }
                return floats;
            }

            private static string[] Strings(string json, string key)
            {
                string[] parts = Elements(json, key);
                for (int i = 0; i < parts.Length; i++) { parts[i] = parts[i].Trim().Trim('"'); }
                return parts;
            }
        }
    }
}
