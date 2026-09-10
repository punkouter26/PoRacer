using System;
using System.Collections.Generic;
using System.Threading;
using Cysharp.Threading.Tasks;
using PoRacer.Agents;
using PoRacer.Models;
using Unity.InferenceEngine;
using Unity.MLAgents.Policies;
using UnityEngine;
using VContainer.Unity;

namespace PoRacer.Systems
{
    /// <summary>
    /// Pays the first race's one-off start-up cost while the menu is up, instead of
    /// on the frame the player presses START.
    ///
    /// WHAT WAS ACTUALLY WRONG. The first race of every session froze for about four
    /// and a half seconds: `fpsMin` of 0.218, 0.215 and 0.223 across three independent
    /// smoke runs on step 0, against nothing worse than 11 fps from step 2 on. Two
    /// plausible culprits were measured and cleared before the right one was found —
    /// building an Inference Engine worker for all eight brains costs 0.32 s, and
    /// running an inference through each to force the backend's Burst jobs to compile
    /// costs the same again. Stage timers in Systems_Spawn named the real one:
    ///
    ///     [SpawnStage] instantiate grid: 4.362 s
    ///
    /// It is prefab instantiation, and it is one-off per prefab TYPE, not per racer:
    /// the same eight prefabs instantiate with no stall at all on the second race of a
    /// session. Loading the asset, building its ArticulationBody chain and standing up
    /// its ML-Agents policy is work that only happens the first time.
    ///
    /// So the warm-up instantiates one of each, holds it still, gives it a couple of
    /// frames to finish standing itself up, and throws it away. Both halves are kept —
    /// the brains are warmed too, because 0.32 s off the countdown is still 0.32 s.
    ///
    /// It is best-effort throughout. Anything that throws warns and moves on: a warm-up
    /// that breaks the game it is meant to smooth would be a poor trade, and all of this
    /// is work the racers would otherwise do for themselves.
    /// </summary>
    public sealed class Systems_Warmup : IStartable, IDisposable
    {
        // Matches every runtime worker in the project: the ML-Agents racers run
        // InferenceDevice.Burst and the Isaac ports build BackendType.CPU explicitly.
        // Both land on the same CPU backend, so warming it once warms it for all.
        private const BackendType WARMUP_BACKEND = BackendType.CPU;

        // Far below any track, so a probe cannot touch the menu's camera, the ground,
        // or another probe. Nothing races here and nothing is looking.
        private static readonly Vector3 ProbePosition = new(0f, -500f, 0f);

        // Frames a probe is left alive. One is not enough: Awake and OnEnable run
        // immediately, but the policy is not stood up until the agent is first asked
        // for a decision, which is the part worth paying for early.
        private const int PROBE_FRAMES = 2;

        private readonly CreatureCatalog _catalog;
        private readonly CancellationTokenSource _cts = new();

        public Systems_Warmup(CreatureCatalog catalog)
        {
            _catalog = catalog;
        }

        /// <summary>
        /// True once every brain and prefab has been through the warm-up, or the attempt
        /// has been abandoned. <see cref="Systems_Spawn"/> waits on this before it builds
        /// a grid.
        ///
        /// Waiting matters as much as warming. Measured: with the warm-up running and
        /// nothing waiting for it, the first race's worst frame went from 4.6 s to
        /// 6.1 s — the work had not moved anywhere, it was just happening alongside the
        /// spawn instead of before it, and the two were fighting for the same frames.
        /// </summary>
        public bool IsComplete { get; private set; }

        public void Start()
        {
            WarmAsync(_cts.Token).Forget();
        }

        public void Dispose()
        {
            _cts.Cancel();
            _cts.Dispose();
        }

        private async UniTaskVoid WarmAsync(CancellationToken token)
        {
            try
            {
                // A frame first: the scope has only just been built and the menu has not
                // drawn yet. Stalling before the player has seen anything is the one
                // place this could make the cold start feel worse rather than better.
                await UniTask.NextFrame(token);

                float startedAt = Time.realtimeSinceStartup;
                // Keyed on the asset references: roster slots share brains and prefabs,
                // and warming the same one twice just doubles what this exists to avoid.
                // Not on GetInstanceID() — Unity 6000.6 made that a compile error.
                var warmedBrains = new HashSet<ModelAsset>();
                var warmedPrefabs = new HashSet<GameObject>();
                int brains = 0;
                int prefabs = 0;

                for (int entryIndex = 0; entryIndex < _catalog.Entries.Count; entryIndex++)
                {
                    CreatureCatalog.CreatureEntry entry = _catalog.Entries[entryIndex];
                    // A brain can exist with no ModelAsset behind it — the MuJoCo racers
                    // read an MLP out of JSON rather than an ONNX graph. Nothing to warm.
                    if (entry.model != null && warmedBrains.Add(entry.model))
                    {
                        WarmBrain(entry.model, entry.id);
                        brains++;
                        await UniTask.NextFrame(token);
                    }
                    if (entry.prefab != null && warmedPrefabs.Add(entry.prefab)
                        && await WarmPrefab(entry, token))
                    {
                        prefabs++;
                    }
                }
                ReportDuration(brains, prefabs, Time.realtimeSinceStartup - startedAt);
            }
            catch (OperationCanceledException)
            {
            }
            finally
            {
                // Set on every exit, cancellation included: a spawner waiting on this
                // must never be left waiting forever because the warm-up gave up.
                IsComplete = true;
            }
        }

        /// <summary>
        /// Loads the graph and runs one inference through it, which is the part that
        /// costs anything: the backend's Burst jobs are compiled the first time they are
        /// scheduled, not when the worker is constructed.
        /// </summary>
        private static void WarmBrain(ModelAsset modelAsset, string creatureId)
        {
            try
            {
                Model model = ModelLoader.Load(modelAsset);
                using var worker = new Worker(model, WARMUP_BACKEND);
                if (!TryGetInputWidth(model, out int inputWidth))
                {
                    return;
                }
                using var input = new Tensor<float>(new TensorShape(1, inputWidth));
                worker.Schedule(input);
                // Completed, not just scheduled: the jobs are async, and returning early
                // would leave the compile to land on some later frame — most likely the
                // first race's, which is the frame this exists to protect.
                if (worker.PeekOutput() is Tensor<float> output)
                {
                    output.CompleteAllPendingOperations();
                }
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"Brain warm-up for '{creatureId}' failed; the first race will "
                               + $"pay for it instead. {exception.Message}");
            }
        }

        /// <summary>
        /// Instantiates one racer of this type far below the world, holds it still for a
        /// couple of frames and destroys it — paying the asset load, the articulation
        /// build and the policy stand-up that otherwise all land on the first grid.
        ///
        /// MuJoCo racers are skipped, and not as an oversight: MjScene is a singleton
        /// that <see cref="Systems_MujocoWorld"/> builds and tears down per race, and an
        /// MjComponent coming to life outside that would claim it. They keep their share
        /// of the first-race cost.
        /// </summary>
        private async UniTask<bool> WarmPrefab(CreatureCatalog.CreatureEntry entry, CancellationToken token)
        {
            if (entry.prefab.GetComponentInChildren<IMujocoCreature>(true) != null)
            {
                return false;
            }
            GameObject probe = null;
            try
            {
                probe = UnityEngine.Object.Instantiate(entry.prefab, ProbePosition, Quaternion.identity);
                probe.name = "Warmup." + entry.id;
                // Held before anything else, so the probe never actuates: it has no
                // RacerView, no arena guard and no ground under it.
                ICreatureAgent agent = FindCreatureAgent(probe);
                if (agent != null)
                {
                    agent.StartHeld = true;
                    agent.MaxStep = 0;
                }
                // The same wiring the spawner does, because standing up the policy is
                // what is being paid for and it is built from these.
                BehaviorParameters behavior = probe.GetComponentInChildren<BehaviorParameters>();
                if (behavior != null && entry.model != null)
                {
                    behavior.Model = entry.model;
                    behavior.BehaviorType = BehaviorType.InferenceOnly;
                    behavior.InferenceDevice = InferenceDevice.Burst;
                }
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"Prefab warm-up for '{entry.id}' failed; the first race will "
                               + $"pay for it instead. {exception.Message}");
            }

            for (int frame = 0; frame < PROBE_FRAMES; frame++)
            {
                await UniTask.NextFrame(token);
            }
            if (probe != null)
            {
                UnityEngine.Object.Destroy(probe);
            }
            // A frame for the destroy to land before the next probe is built, so two
            // creatures are never alive down there at once.
            await UniTask.NextFrame(token);
            return true;
        }

        /// <summary>First ICreatureAgent under the instance, whatever MonoBehaviour implements it.</summary>
        private static ICreatureAgent FindCreatureAgent(GameObject instance)
        {
            MonoBehaviour[] behaviours = instance.GetComponentsInChildren<MonoBehaviour>(true);
            for (int index = 0; index < behaviours.Length; index++)
            {
                if (behaviours[index] is ICreatureAgent creature)
                {
                    return creature;
                }
            }
            return null;
        }

        /// <summary>
        /// Width of the model's first input, when the graph declares a static one.
        ///
        /// Every brain in the roster takes a single flat (1, obs) observation vector, so
        /// this is the whole of the shape that matters. A model with a dynamic or
        /// unranked input is skipped rather than guessed at — feeding a wrong shape would
        /// throw, and the point here is to be free.
        /// </summary>
        private static bool TryGetInputWidth(Model model, out int inputWidth)
        {
            inputWidth = 0;
            if (model.inputs == null || model.inputs.Count == 0)
            {
                return false;
            }
            DynamicTensorShape shape = model.inputs[0].shape;
            if (!shape.IsStatic())
            {
                return false;
            }
            TensorShape staticShape = shape.ToTensorShape();
            if (staticShape.rank < 1)
            {
                return false;
            }
            inputWidth = staticShape[staticShape.rank - 1];
            return inputWidth > 0;
        }

        [System.Diagnostics.Conditional("UNITY_EDITOR")]
        private static void ReportDuration(int brainCount, int prefabCount, float seconds)
        {
            Debug.Log($"[Warmup] {brainCount} brain(s) and {prefabCount} prefab(s) in {seconds:0.00} s; "
                    + "the first race would otherwise have paid this when START was pressed.");
        }
    }
}
