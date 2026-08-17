using System;
using System.Collections.Generic;
using Cysharp.Threading.Tasks;
using MessagePipe;
using PoRacer.Agents;
using PoRacer.Models;
using PoRacer.Views;
using System.Threading;
using Unity.MLAgents.Policies;
using UnityEngine;
using VContainer.Unity;

namespace PoRacer.Systems
{
    /// <summary>
    /// Spawns racers from the creature catalog per the RaceConfigModel counts,
    /// in grid rows behind the start line (marathon start), then runs the endless
    /// race loop: race finished -> pause -> despawn -> respawn -> next race.
    /// BeginRacing() is called by the menu; RequestMenu() stops the loop.
    /// A racer whose .onnx model is missing is skipped with a warning (never crashes).
    /// </summary>
    public sealed class Systems_Spawn : IStartable, IDisposable
    {
        private const float SECONDS_BETWEEN_RACES = 5f;
        private const int DECISION_PERIOD = 5;

        // Visible quirks: same brain, mild physics tweak, HUD badge + name prefix.
        private readonly struct QuirkDef
        {
            public readonly string Prefix;
            public readonly string Tag;
            public readonly float Power;
            public readonly float MassScale;
            public readonly float Weight;
            public readonly Color Badge;

            public QuirkDef(string prefix, string tag, float power, float massScale, float weight, Color badge)
            {
                Prefix = prefix;
                Tag = tag;
                Power = power;
                MassScale = massScale;
                Weight = weight;
                Badge = badge;
            }
        }

        private static readonly QuirkDef[] Quirks =
        {
            new(string.Empty, string.Empty, 1f, 1f, 0.30f, Color.clear),
            new("Mighty ", "MIGHTY", 1.06f, 1f, 0.16f, new Color(1f, 0.45f, 0.2f)),
            new("Sleepy ", "SLEEPY", 0.94f, 1f, 0.16f, new Color(0.4f, 0.6f, 1f)),
            new("Turbo ", "TURBO", 1.12f, 1f, 0.10f, new Color(1f, 0.85f, 0.2f)),
            new("Heavy ", "HEAVY", 1f, 1.12f, 0.14f, new Color(0.65f, 0.65f, 0.7f)),
            new("Feather ", "FEATHER", 1f, 0.9f, 0.14f, new Color(0.95f, 0.95f, 0.85f))
        };

        // Roulette map: kinds the trained brains already know from the curriculum.
        private static readonly TrackKind[] RouletteKinds =
        {
            TrackKind.Flat, TrackKind.Bumps, TrackKind.Walls, TrackKind.Lumpy, TrackKind.Swamp
        };
        // Golden-ratio hue stepping spreads racer tints evenly around the wheel.
        private const float TINT_HUE_STEP = 0.61803f;

        private static readonly int BaseColorId = Shader.PropertyToID("_BaseColor");
        private static readonly int LegacyColorId = Shader.PropertyToID("_Color");
        private static readonly int SmoothnessId = Shader.PropertyToID("_Smoothness");
        private const int GRID_COLUMNS = 10;
        private const float GRID_X_SPACING = 2f;
        private const float GRID_ROW_SPACING = 1.6f;
        // Big fields start as a tower: keep a small footprint and stack layers
        // upward, so the start is a glorious collapsing pile.
        private const int STACK_THRESHOLD = 30;
        private const int STACK_FOOTPRINT_ROWS = 3;
        private const float STACK_LAYER_HEIGHT = 1.5f;
        private const float STACK_JITTER = 0.35f;

        private readonly CreatureCatalog _catalog;
        private readonly RaceConfigModel _config;
        private readonly RaceTrackView _track;
        private readonly Systems_Race _race;
        private readonly Systems_CameraDirector _cameraDirector;
        private readonly RaceModel _raceModel;
        private readonly Systems_TrackBuilder _trackBuilder;
        private readonly System.Random _rng = new();
        private readonly IDisposable _subscription;
        private CancellationTokenSource _cts = new();
        private readonly List<GameObject> _spawned = new();
        private readonly List<Transform> _racerRoots = new();
        private readonly List<int> _nameOrder = new();
        private readonly MaterialPropertyBlock _tintBlock = new();
        private bool _racingLoopActive;
        // Incremented on every BeginRacing/RequestMenu; async chains capture it and
        // bail out after each await if a newer session has superseded them.
        private int _generation;
        private TrackKind _currentTrack = TrackKind.Flat;

        public Systems_Spawn(
            CreatureCatalog catalog,
            RaceConfigModel config,
            RaceTrackView track,
            Systems_Race race,
            Systems_CameraDirector cameraDirector,
            RaceModel raceModel,
            Systems_TrackBuilder trackBuilder,
            ISubscriber<RaceFinishedMessage> raceFinished)
        {
            _catalog = catalog;
            _config = config;
            _track = track;
            _race = race;
            _cameraDirector = cameraDirector;
            _raceModel = raceModel;
            _trackBuilder = trackBuilder;
            _subscription = raceFinished.Subscribe(OnRaceFinished);
        }

        public void Start()
        {
            // Every raceable creature enters with 1 racer by default; the menu
            // shows those counts pre-selected on first launch.
            for (int entryIndex = 0; entryIndex < _catalog.Entries.Count; entryIndex++)
            {
                CreatureCatalog.CreatureEntry entry = _catalog.Entries[entryIndex];
                if (entry.prefab != null && entry.model != null)
                {
                    _config.SetCount(entry.id, 1);
                }
            }
            _config.MenuVisible = true;
            _config.NotifyChanged();
        }

        public void BeginRacing()
        {
            if (_config.TotalCount() == 0)
            {
                Debug.LogWarning("No racers selected; staying in menu.");
                return;
            }
            _config.MenuVisible = false;
            _config.NotifyChanged();
            _racingLoopActive = true;
            SpawnAndStartRace(++_generation, _cts.Token).Forget();
        }

        public void RequestMenu()
        {
            _racingLoopActive = false;
            _generation++;
            _cts.Cancel();
            _cts.Dispose();
            _cts = new CancellationTokenSource();
            Despawn();
            _raceModel.CountdownValue = 0;
            _cameraDirector.SetTargets(System.Array.Empty<Transform>());
            _race.AbortRace();
            _config.MenuVisible = true;
            _config.NotifyChanged();
        }

        public void Dispose()
        {
            _subscription.Dispose();
            _cts.Cancel();
            _cts.Dispose();
        }

        private void OnRaceFinished(RaceFinishedMessage message)
        {
            if (_racingLoopActive)
            {
                RestartAfterDelay(_generation).Forget();
            }
        }

        private bool IsCurrent(int generation) => _racingLoopActive && generation == _generation;

        private async UniTaskVoid RestartAfterDelay(int generation)
        {
            await UniTask.Delay(TimeSpan.FromSeconds(SECONDS_BETWEEN_RACES), cancellationToken: _cts.Token);
            if (!IsCurrent(generation))
            {
                return;
            }
            Despawn();
            // The podium had its 5 s; from here the old roster is history. Clearing
            // it keeps the HUD from listing last race's results over the new grid.
            _raceModel.ClearRacers();
            // Destroy() is deferred to end of frame; spawning immediately would
            // overlap new worms with old colliders and explode the physics.
            await UniTask.NextFrame(_cts.Token);
            if (!IsCurrent(generation))
            {
                return;
            }
            await SpawnAndStartRace(generation, _cts.Token);
        }

        private async UniTask SpawnAndStartRace(int generation, CancellationToken token)
        {
            Systems_MapCatalog.MapEntry map = Systems_MapCatalog.Get(_config.SelectedMapIndex);
            TrackKind rolledKind = map.Kind;
            TrackFeatures rolledFeatures = map.Features;
            string trackName = map.DisplayName;
            if (map.Randomize)
            {
                // Roulette: roll terrain and hazards fresh for every race.
                rolledKind = RouletteKinds[_rng.Next(RouletteKinds.Length)];
                rolledFeatures = TrackFeatures.None;
                if (rolledKind != TrackKind.Swamp && _rng.Next(2) == 0)
                {
                    rolledFeatures |= TrackFeatures.MudPits;
                }
                if (_rng.Next(2) == 0)
                {
                    rolledFeatures |= TrackFeatures.BoostPads;
                }
                if (_rng.Next(3) == 0)
                {
                    rolledFeatures |= TrackFeatures.Gusts;
                }
                if (rolledKind != TrackKind.Swamp && _rng.Next(3) == 0)
                {
                    rolledFeatures |= TrackFeatures.Gates;
                }
                trackName = $"Roulette: {rolledKind}";
            }
            _currentTrack = rolledKind;
            _raceModel.TrackName = trackName;
            // The finish line (trigger + agent goal) moves to the map's length so
            // race distance is a per-map design knob, not a scene constant.
            if (_track.FinishLine != null)
            {
                Vector3 finishPosition = _track.FinishLine.position;
                finishPosition.z = map.LengthMeters - 2f;
                _track.FinishLine.position = finishPosition;
            }
            if (_track.TrackRoot != null)
            {
                float finishZ = _track.FinishLine != null ? _track.FinishLine.position.z : -1f;
                // The ground must reach past the last grid row: big rosters spawn
                // many rows deep behind the start line. Stacked starts keep the
                // small tower footprint instead.
                int totalRequested = _config.TotalCount();
                int gridRows = totalRequested > STACK_THRESHOLD
                    ? STACK_FOOTPRINT_ROWS
                    : (totalRequested + GRID_COLUMNS - 1) / GRID_COLUMNS;
                float backMargin = Mathf.Max(7f, gridRows * GRID_ROW_SPACING + 4f);
                _trackBuilder.Build(_currentTrack, _track.TrackRoot, width: 24f, length: map.LengthMeters, _rng,
                    decorate: true, finishZ: finishZ, features: rolledFeatures, backMargin: backMargin);
                // Freshly built colliders must exist before racers land on them.
                await UniTask.NextFrame(token);
                if (!IsCurrent(generation))
                {
                    return;
                }
            }

            var racers = new List<RacerState>();
            var pendingQuirks = new List<(GameObject instance, float power, float massScale)>();
            _racerRoots.Clear();
            RacerNames.Shuffle(_rng, _nameOrder);
            Vector3 gridOrigin = _track.SpawnPoints.Count > 0
                ? _track.SpawnPoints[0].parent.position
                : Vector3.zero;
            if (_track.FinishLine != null)
            {
                _raceModel.TrackLengthMeters = Mathf.Max(1f, _track.FinishLine.position.z - gridOrigin.z);
            }

            int gridIndex = 0;
            for (int entryIndex = 0; entryIndex < _catalog.Entries.Count; entryIndex++)
            {
                CreatureCatalog.CreatureEntry entry = _catalog.Entries[entryIndex];
                int requested = _config.GetCount(entry.id);
                if (requested <= 0)
                {
                    continue;
                }
                bool scripted = _config.UseScriptedBrains;
                if (entry.prefab == null || (entry.model == null && !scripted))
                {
                    Debug.LogWarning($"Creature '{entry.id}' has no trained brain yet; skipping its {requested} racers.");
                    continue;
                }
                for (int racerIndex = 0; racerIndex < requested; racerIndex++)
                {
                    // Spread large spawns over frames: 800 articulated bodies in one
                    // frame is a visible hitch, and it repeats every race.
                    if (gridIndex > 0 && gridIndex % 25 == 0)
                    {
                        await UniTask.NextFrame(token);
                        if (!IsCurrent(generation))
                        {
                            Despawn();
                            return;
                        }
                    }
                    // Tower start for big fields: the same footprint repeats in
                    // layers going up, with jitter so the pile topples, not balances.
                    bool stacked = _config.TotalCount() > STACK_THRESHOLD;
                    int layerSize = GRID_COLUMNS * STACK_FOOTPRINT_ROWS;
                    int layer = stacked ? gridIndex / layerSize : 0;
                    int flatIndex = stacked ? gridIndex % layerSize : gridIndex;
                    int column = flatIndex % GRID_COLUMNS;
                    int row = flatIndex / GRID_COLUMNS;
                    float localZ = -row * GRID_ROW_SPACING;
                    float localX = (column - (GRID_COLUMNS - 1) * 0.5f) * GRID_X_SPACING;
                    if (layer > 0)
                    {
                        localX += ((float)_rng.NextDouble() - 0.5f) * 2f * STACK_JITTER;
                        localZ += ((float)_rng.NextDouble() - 0.5f) * 2f * STACK_JITTER;
                    }
                    // Small extra drop height so nobody is born intersecting the ground.
                    Vector3 position = gridOrigin + new Vector3(
                        localX,
                        Systems_TrackBuilder.SurfaceHeight(_currentTrack, localX, localZ) + entry.spawnHeight + 0.05f
                            + layer * STACK_LAYER_HEIGHT,
                        localZ);

                    GameObject instance = UnityEngine.Object.Instantiate(entry.prefab, position, Quaternion.identity);
                    // Generation-scoped so an orphan from a superseded spawn chain can
                    // never collide with a live racer's ID in RaceModel.
                    string racerId = $"{entry.id}#{generation}.{gridIndex + 1}";
                    instance.name = racerId;

                    var agent = instance.GetComponentInChildren<Unity.MLAgents.Agent>() as ICreatureAgent;
                    BehaviorParameters behavior = instance.GetComponentInChildren<BehaviorParameters>();
                    if (agent == null || behavior == null)
                    {
                        Debug.LogWarning($"Creature '{entry.id}' prefab lacks an ICreatureAgent; skipping.");
                        UnityEngine.Object.Destroy(instance);
                        continue;
                    }
                    if (scripted || entry.model == null)
                    {
                        // Coded gait instead of a brain: Heuristic() drives the joints.
                        behavior.BehaviorType = BehaviorType.HeuristicOnly;
                    }
                    else
                    {
                        behavior.Model = entry.model;
                        behavior.BehaviorType = BehaviorType.InferenceOnly;
                        behavior.InferenceDevice = InferenceDevice.Burst;
                    }
                    agent.MaxStep = 0;
                    agent.SetGoal(_track.FinishLine);

                    var decisionRequester = instance.GetComponentInChildren<Unity.MLAgents.DecisionRequester>();
                    if (decisionRequester != null)
                    {
                        decisionRequester.DecisionStep = gridIndex % DECISION_PERIOD;
                    }

                    // Quirk: scale joint drives and body mass so this racer runs a
                    // touch differently than its siblings with the same brain.
                    // Applied one frame later — a joint written in the frame it was
                    // instantiated rejects the drive as non-finite.
                    QuirkDef quirk = PickQuirk();
                    // Small jitter so same-quirk siblings still differ a touch.
                    float quirkPower = quirk.Power * (0.99f + (float)_rng.NextDouble() * 0.02f);
                    pendingQuirks.Add((instance, quirkPower, quirk.MassScale));
                    string funName = RacerNames.Get(_nameOrder, gridIndex);

                    // Connective limb visuals must exist before tinting so the
                    // links pick up this racer's color along with its parts.
                    instance.AddComponent<BodyLinkView>();

                    // Unique tint per racer via property block: shared material
                    // stays shared, so batching is not broken by material clones.
                    // Alternating light/dark segments plus a touch of gloss give
                    // the primitive bodies visible form under the sun.
                    Color tint = Color.HSVToRGB(gridIndex * TINT_HUE_STEP % 1f, 0.6f, 1f);
                    Color darkTint = new Color(tint.r * 0.78f, tint.g * 0.78f, tint.b * 0.78f);
                    Renderer[] tintRenderers = instance.GetComponentsInChildren<Renderer>();
                    for (int rendererIndex = 0; rendererIndex < tintRenderers.Length; rendererIndex++)
                    {
                        Color segmentTint = rendererIndex % 2 == 0 ? tint : darkTint;
                        _tintBlock.Clear();
                        _tintBlock.SetColor(BaseColorId, segmentTint);
                        _tintBlock.SetColor(LegacyColorId, segmentTint);
                        _tintBlock.SetFloat(SmoothnessId, 0.5f);
                        tintRenderers[rendererIndex].SetPropertyBlock(_tintBlock);
                    }

                    RacerView view = instance.AddComponent<RacerView>();
                    float finishZ = _track.FinishLine != null ? _track.FinishLine.position.z : float.PositiveInfinity;
                    view.Initialize(racerId, _race, position.z, agent, finishZ);
                    instance.AddComponent<DustTrailView>();
                    instance.AddComponent<CreatureAudioView>();
                    // After tinting on purpose: the eyes keep their own colors.
                    instance.AddComponent<EyesView>();
                    instance.AddComponent<SkidMarkView>().Initialize(_currentTrack);

                    _spawned.Add(instance);
                    _racerRoots.Add(instance.transform);
                    racers.Add(new RacerState
                    {
                        RacerId = racerId,
                        CreatureId = entry.id,
                        DisplayName = $"{quirk.Prefix}{funName} the {entry.displayName}",
                        Status = RacerStatus.Racing,
                        Tint = tint,
                        TintHex = ColorUtility.ToHtmlStringRGB(tint),
                        QuirkTag = quirk.Tag,
                        QuirkColor = quirk.Badge
                    });
                    gridIndex++;
                }
            }

            _cameraDirector.SetTargets(_racerRoots);
            // Registered after SetTargets, which clears the id registry.
            for (int racerIndex = 0; racerIndex < racers.Count; racerIndex++)
            {
                _cameraDirector.RegisterRacer(racers[racerIndex].RacerId, _racerRoots[racerIndex]);
            }
            if (racers.Count == 0)
            {
                Debug.LogWarning("No valid racers could be spawned; returning to menu.");
                RequestMenu();
                return;
            }

            // One frame so physics initializes every articulation, then the power
            // quirks apply cleanly.
            await UniTask.NextFrame(token);
            if (!IsCurrent(generation))
            {
                Despawn();
                return;
            }
            for (int quirkIndex = 0; quirkIndex < pendingQuirks.Count; quirkIndex++)
            {
                ApplyQuirk(pendingQuirks[quirkIndex].instance, pendingQuirks[quirkIndex].power,
                    pendingQuirks[quirkIndex].massScale);
            }

            // 3-2-1 countdown: the grid settles physically while the HUD counts
            // and the audio director beeps along (it watches CountdownValue).
            for (int countdown = 3; countdown >= 1; countdown--)
            {
                _raceModel.CountdownValue = countdown;
                await UniTask.Delay(TimeSpan.FromSeconds(0.8), cancellationToken: token);
                if (!IsCurrent(generation))
                {
                    _raceModel.CountdownValue = 0;
                    return;
                }
            }
            _raceModel.CountdownValue = 0;
            _race.StartRace(racers);
        }

        private QuirkDef PickQuirk()
        {
            float totalWeight = 0f;
            for (int quirkIndex = 0; quirkIndex < Quirks.Length; quirkIndex++)
            {
                totalWeight += Quirks[quirkIndex].Weight;
            }
            float roll = (float)_rng.NextDouble() * totalWeight;
            for (int quirkIndex = 0; quirkIndex < Quirks.Length; quirkIndex++)
            {
                roll -= Quirks[quirkIndex].Weight;
                if (roll <= 0f)
                {
                    return Quirks[quirkIndex];
                }
            }
            return Quirks[0];
        }

        private static void ApplyQuirk(GameObject instance, float quirkPower, float massScale)
        {
            if (instance == null)
            {
                return;
            }
            ArticulationBody[] bodies = instance.GetComponentsInChildren<ArticulationBody>();
            for (int bodyIndex = 0; bodyIndex < bodies.Length; bodyIndex++)
            {
                if (massScale != 1f)
                {
                    bodies[bodyIndex].mass *= massScale;
                }
                ArticulationDrive drive = bodies[bodyIndex].xDrive;
                // Guard the PRODUCT, not the input: locked joints author
                // float.MaxValue budgets, and MaxValue * 1.05 overflows to
                // infinity, which the physics engine rejects wholesale.
                float scaledStiffness = drive.stiffness * quirkPower;
                float scaledForceLimit = drive.forceLimit * quirkPower;
                if (float.IsFinite(scaledStiffness))
                {
                    drive.stiffness = scaledStiffness;
                }
                if (float.IsFinite(scaledForceLimit))
                {
                    drive.forceLimit = scaledForceLimit;
                }
                bodies[bodyIndex].xDrive = drive;
            }
            // Fatigue captures its full-power baseline lazily; the quirked drives
            // must be what it captures, not the prefab's authored values.
            var agent = instance.GetComponentInChildren<Unity.MLAgents.Agent>() as ICreatureAgent;
            if (agent != null)
            {
                agent.NotifyDrivesChanged();
            }
        }

        private void Despawn()
        {
            for (int spawnedIndex = 0; spawnedIndex < _spawned.Count; spawnedIndex++)
            {
                if (_spawned[spawnedIndex] != null)
                {
                    UnityEngine.Object.Destroy(_spawned[spawnedIndex]);
                }
            }
            _spawned.Clear();
            _racerRoots.Clear();
        }
    }
}
