using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Creature.MojucuBoy;
using Mujoco;
using UnityEngine;
#if CREATURE_HAS_INFERENCE
using Unity.InferenceEngine;
#endif

namespace CreatureEditor
{
    /// <summary>
    /// Phase 5 verification, in two parts.
    ///
    /// ReplayReference() feeds the stored observations from mujoco_reference.json
    /// through Unity's Inference Engine and compares the actions against the ones
    /// Python produced. That isolates the INFERENCE: it answers "does the exported
    /// graph compute the same function here?" with no physics involved, so a
    /// failure cannot be confused with a dynamics difference.
    ///
    /// Evaluate() then runs the policy closed-loop against Unity's own MuJoCo and
    /// reports the same metrics gate4_eval.py reports, for a side-by-side.
    ///
    /// Both drive MuJoCo directly rather than through play mode: the controller's
    /// MonoBehaviour lifecycle adds nothing to what is being measured, and a
    /// CLI-driven play mode is flaky. The observation, the action mapping and the
    /// inference call are all the same code the runtime controller uses.
    /// </summary>
    public static class MojucuBoyEvalHarness
    {
        private const string RIG_JSON = "Assets/Agents/MojucuBoy_v01/mojucuboy_rig.json";
        private const string ONNX_PATH = "Assets/Agents/MojucuBoy_v01/MojucuBoy_v01.onnx";
        private const int EPISODE_STEPS = 1000;
        private const int DECIMATION = 4;
        private const float ACTION_SCALE = MojucuBoyController.ACTION_SCALE;

        public static string ReplayReference(string referencePath)
        {
#if !CREATURE_HAS_INFERENCE
            return "ABORT: built without CREATURE_HAS_INFERENCE.";
#else
            string json = File.ReadAllText(referencePath);
            List<float[]> observations = Vectors(json, "obs");
            List<float[]> expected = Vectors(json, "action");
            if (observations.Count == 0)
            {
                return "ABORT: reference file has no steps.";
            }

            var asset = UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>(ONNX_PATH);
            if (asset == null)
            {
                return $"ABORT: {ONNX_PATH} not found.";
            }
            using var worker = new Worker(ModelLoader.Load(asset), BackendType.CPU);
            using var input = new Tensor<float>(new TensorShape(1, MojucuBoyObservation.OBS_SIZE));

            double worst = 0.0;
            int worstStep = -1, worstIndex = -1;
            for (int step = 0; step < observations.Count; step++)
            {
                input.Upload(observations[step]);
                worker.Schedule(input);
                var output = worker.PeekOutput() as Tensor<float>;
                output.CompleteAllPendingOperations();
                for (int i = 0; i < MojucuBoyObservation.ACTION_SIZE; i++)
                {
                    double delta = Math.Abs(output[0, i] - expected[step][i]);
                    if (delta > worst)
                    {
                        worst = delta;
                        worstStep = step;
                        worstIndex = i;
                    }
                }
            }
            return $"replayed {observations.Count} reference steps; "
                 + $"max |delta| = {worst:E3} at step {worstStep} action {worstIndex}";
#endif
        }

        public static string Evaluate(int episodes, string outPath)
        {
#if !CREATURE_HAS_INFERENCE
            return "ABORT: built without CREATURE_HAS_INFERENCE.";
#else
            // Do NOT reach for MjScene.Instance here. In edit mode a scene-authored
            // MjScene has not had Awake() run, so the static _instance is still null
            // -- and the Instance getter throws in exactly that case rather than
            // adopting the component it can plainly see. Find it and let it claim
            // the singleton itself.
            MjScene scene = UnityEngine.Object.FindAnyObjectByType<MjScene>();
            if (scene == null)
            {
                scene = MjScene.Instance;   // none in the scene: safe to create one
            }
            else if (!MjScene.InstanceExists)
            {
                scene.Awake();
            }
            scene.CreateScene(false);

            var rigAsset = UnityEditor.AssetDatabase.LoadAssetAtPath<TextAsset>(RIG_JSON);
            MojucuBoyRig rig = MojucuBoyRig.Parse(rigAsset.text);
            var asset = UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>(ONNX_PATH);
            if (asset == null)
            {
                return $"ABORT: {ONNX_PATH} not found.";
            }
            using var worker = new Worker(ModelLoader.Load(asset), BackendType.CPU);
            using var input = new Tensor<float>(new TensorShape(1, MojucuBoyObservation.OBS_SIZE));
            return Run(scene, rig, rigAsset.text, worker, input, episodes, outPath);
#endif
        }

#if CREATURE_HAS_INFERENCE
        private static unsafe string Run(MjScene scene, MojucuBoyRig rig, string rigJson,
                                         Worker worker, Tensor<float> input,
                                         int episodes, string outPath)
        {
            MujocoLib.mjModel_* model = scene.Model;
            MujocoLib.mjData_* data = scene.Data;
            int nq = (int)model->nq;
            int nv = (int)model->nv;
            int n = rig.ActuatorOrder.Length;

            int rootBody = MujocoLib.mj_name2id(model, (int)MujocoLib.mjtObj.mjOBJ_BODY, rig.RootBody);
            var qposAddr = new int[n];
            var dofAddr = new int[n];
            var actAddr = new int[n];
            for (int i = 0; i < n; i++)
            {
                int jid = MujocoLib.mj_name2id(model, (int)MujocoLib.mjtObj.mjOBJ_JOINT, rig.ActuatorOrder[i]);
                qposAddr[i] = model->jnt_qposadr[jid];
                dofAddr[i] = model->jnt_dofadr[jid];
                actAddr[i] = MujocoLib.mj_name2id(model, (int)MujocoLib.mjtObj.mjOBJ_ACTUATOR,
                                                  "act_" + rig.ActuatorOrder[i]);
            }

            float[] stanceQpos = ParseFloats(rigJson, "stance_qpos");

            var obs = new float[MojucuBoyObservation.OBS_SIZE];
            var action = new float[MojucuBoyObservation.ACTION_SIZE];
            var lengths = new List<int>();
            var speeds = new List<double>();
            int survivors = 0;
            var random = new System.Random(4242);

            for (int episode = 0; episode < episodes; episode++)
            {
                MujocoLib.mj_resetData(model, data);
                for (int i = 0; i < nq && i < stanceQpos.Length; i++)
                {
                    data->qpos[i] = stanceQpos[i];
                }
                // Same reset distribution as training: random yaw, small joint noise.
                double yaw = (random.NextDouble() * 2.0 - 1.0) * Math.PI;
                data->qpos[3] = Math.Cos(yaw / 2.0);
                data->qpos[4] = 0.0;
                data->qpos[5] = 0.0;
                data->qpos[6] = Math.Sin(yaw / 2.0);
                for (int i = 0; i < n; i++)
                {
                    double noise = (random.NextDouble() * 2.0 - 1.0) * 0.10;
                    double value = stanceQpos[qposAddr[i]] + noise;
                    data->qpos[qposAddr[i]] = Math.Min(Math.Max(value, rig.RangeLo[i]), rig.RangeHi[i]);
                }
                for (int i = 0; i < nv; i++) { data->qvel[i] = 0.0; }
                Array.Clear(action, 0, action.Length);
                MujocoLib.mj_forward(model, data);

                float commandHeading = (float)(yaw + (random.NextDouble() * 2.0 - 1.0) * 0.6);
                double cosH = Math.Cos(commandHeading), sinH = Math.Sin(commandHeading);

                int step = 0;
                double speedSum = 0.0;
                for (; step < EPISODE_STEPS; step++)
                {
                    // One policy evaluation per outer step; the DECIMATION physics
                    // steps below make that outer step 0.02 s, the trained rate.
                    MojucuBoyObservation.Build(data, rootBody, qposAddr, dofAddr,
                                         commandHeading, 1.5f, action, obs);
                    input.Upload(obs);
                    worker.Schedule(input);
                    var output = worker.PeekOutput() as Tensor<float>;
                    output.CompleteAllPendingOperations();
                    for (int i = 0; i < n; i++)
                    {
                        action[i] = Mathf.Clamp(output[0, i], -1f, 1f);
                    }
                    for (int i = 0; i < n; i++)
                    {
                        float half = 0.5f * (rig.RangeHi[i] - rig.RangeLo[i]);
                        float target = rig.Stance[i] + ACTION_SCALE * half * action[i];
                        data->ctrl[actAddr[i]] = Mathf.Clamp(target, rig.RangeLo[i], rig.RangeHi[i]);
                    }
                    for (int sub = 0; sub < DECIMATION; sub++)
                    {
                        MujocoLib.mj_step(model, data);
                    }

                    speedSum += data->qvel[0] * cosH + data->qvel[1] * sinH;

                    double height = data->qpos[2];
                    double* xmat = data->xmat + 9 * rootBody;
                    // xmat[8] is R[2][2], +1 when upright. Negating it here would
                    // read -1 for a perfectly upright racer and terminate instantly.
                    double upright = xmat[8];
                    if (height < 0.45 || upright < 0.30)
                    {
                        step++;
                        break;
                    }
                }
                lengths.Add(step);
                speeds.Add(speedSum / Math.Max(step, 1));
                if (step >= EPISODE_STEPS) { survivors++; }
            }

            double meanLength = 0, meanSpeed = 0;
            foreach (int l in lengths) { meanLength += l; }
            foreach (double s in speeds) { meanSpeed += s; }
            meanLength /= lengths.Count;
            meanSpeed /= speeds.Count;
            lengths.Sort();
            double median = lengths[lengths.Count / 2];

            var sb = new StringBuilder();
            sb.Append("{\n  \"episodes\": ").Append(episodes)
              .Append(",\n  \"mean_episode_length\": ").Append(Fmt(meanLength))
              .Append(",\n  \"median_episode_length\": ").Append(Fmt(median))
              .Append(",\n  \"mean_forward_speed\": ").Append(Fmt(meanSpeed))
              .Append(",\n  \"survivors\": ").Append(survivors)
              .Append(",\n  \"survival_rate\": ").Append(Fmt((double)survivors / episodes))
              .Append("\n}\n");
            File.WriteAllText(outPath, sb.ToString());
            return $"episodes {episodes}  meanLen {meanLength:F1}  medianLen {median:F0}  "
                 + $"meanSpeed {meanSpeed:F3} m/s  survivors {survivors}/{episodes}  -> {outPath}";
        }
#endif

        private static string Fmt(double v) => v.ToString("R", CultureInfo.InvariantCulture);

        private static float[] ParseFloats(string json, string key)
        {
            int at = json.IndexOf("\"" + key + "\"", StringComparison.Ordinal);
            int start = json.IndexOf('[', at);
            int end = json.IndexOf(']', start);
            string[] parts = json.Substring(start + 1, end - start - 1).Split(',');
            var values = new float[parts.Length];
            for (int i = 0; i < parts.Length; i++)
            {
                values[i] = float.Parse(parts[i].Trim(), CultureInfo.InvariantCulture);
            }
            return values;
        }

        /// <summary>Pull every "key": [ ... ] array out of the reference file, in order.</summary>
        private static List<float[]> Vectors(string json, string key)
        {
            var results = new List<float[]>();
            string token = "\"" + key + "\"";
            int at = json.IndexOf(token, StringComparison.Ordinal);
            while (at >= 0)
            {
                int start = json.IndexOf('[', at);
                int end = json.IndexOf(']', start);
                string[] parts = json.Substring(start + 1, end - start - 1).Split(',');
                var values = new float[parts.Length];
                for (int i = 0; i < parts.Length; i++)
                {
                    values[i] = float.Parse(parts[i].Trim(), CultureInfo.InvariantCulture);
                }
                results.Add(values);
                at = json.IndexOf(token, end, StringComparison.Ordinal);
            }
            return results;
        }
    }
}
