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
        private const int GRID_COLUMNS = 10;
        private const float GRID_X_SPACING = 2f;
        private const float GRID_ROW_SPACING = 1.6f;

        private readonly CreatureCatalog _catalog;
        private readonly RaceConfigModel _config;
        private readonly RaceTrackView _track;
        private readonly Systems_Race _race;
        private readonly Systems_CameraDirector _cameraDirector;
        private readonly RaceModel _raceModel;
        private readonly Systems_TrackBuilder _trackBuilder;
        private readonly System.Random _rng = new();
        private readonly IDisposable _subscription;
        private readonly CancellationTokenSource _cts = new();
        private readonly List<GameObject> _spawned = new();
        private readonly List<Transform> _racerRoots = new();
        private bool _racingLoopActive;
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
            SpawnAndStartRace(_cts.Token).Forget();
        }

        public void RequestMenu()
        {
            _racingLoopActive = false;
            Despawn();
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
                RestartAfterDelay().Forget();
            }
        }

        private async UniTaskVoid RestartAfterDelay()
        {
            await UniTask.Delay(TimeSpan.FromSeconds(SECONDS_BETWEEN_RACES), cancellationToken: _cts.Token);
            if (!_racingLoopActive)
            {
                return;
            }
            Despawn();
            // Destroy() is deferred to end of frame; spawning immediately would
            // overlap new worms with old colliders and explode the physics.
            await UniTask.NextFrame(_cts.Token);
            if (!_racingLoopActive)
            {
                return;
            }
            await SpawnAndStartRace(_cts.Token);
        }

        private async UniTask SpawnAndStartRace(CancellationToken token)
        {
            _currentTrack = Systems_TrackBuilder.Roll(_rng);
            _raceModel.TrackName = _currentTrack.ToString();
            if (_track.TrackRoot != null)
            {
                _trackBuilder.Build(_currentTrack, _track.TrackRoot, width: 24f, length: 22f, _rng);
                // Freshly built colliders must exist before racers land on them.
                await UniTask.NextFrame(token);
                if (!_racingLoopActive)
                {
                    return;
                }
            }

            var racers = new List<RacerState>();
            _racerRoots.Clear();
            Vector3 gridOrigin = _track.SpawnPoints.Count > 0
                ? _track.SpawnPoints[0].parent.position
                : Vector3.zero;

            int gridIndex = 0;
            for (int entryIndex = 0; entryIndex < _catalog.Entries.Count; entryIndex++)
            {
                CreatureCatalog.CreatureEntry entry = _catalog.Entries[entryIndex];
                int requested = _config.GetCount(entry.id);
                if (requested <= 0)
                {
                    continue;
                }
                if (entry.prefab == null || entry.model == null)
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
                        if (!_racingLoopActive)
                        {
                            Despawn();
                            return;
                        }
                    }
                    int column = gridIndex % GRID_COLUMNS;
                    int row = gridIndex / GRID_COLUMNS;
                    float localZ = -row * GRID_ROW_SPACING;
                    Vector3 position = gridOrigin + new Vector3(
                        (column - (GRID_COLUMNS - 1) * 0.5f) * GRID_X_SPACING,
                        Systems_TrackBuilder.SurfaceHeight(_currentTrack, localZ) + entry.spawnHeight,
                        localZ);

                    GameObject instance = UnityEngine.Object.Instantiate(entry.prefab, position, Quaternion.identity);
                    string racerId = $"{entry.id}#{gridIndex + 1}";
                    instance.name = racerId;

                    var agent = instance.GetComponentInChildren<Unity.MLAgents.Agent>() as ICreatureAgent;
                    BehaviorParameters behavior = instance.GetComponentInChildren<BehaviorParameters>();
                    if (agent == null || behavior == null)
                    {
                        Debug.LogWarning($"Creature '{entry.id}' prefab lacks an ICreatureAgent; skipping.");
                        UnityEngine.Object.Destroy(instance);
                        continue;
                    }
                    behavior.Model = entry.model;
                    behavior.BehaviorType = BehaviorType.InferenceOnly;
                    behavior.InferenceDevice = InferenceDevice.Burst;
                    agent.MaxStep = 0;
                    agent.SetGoal(_track.FinishLine);

                    var decisionRequester = instance.GetComponentInChildren<Unity.MLAgents.DecisionRequester>();
                    if (decisionRequester != null)
                    {
                        decisionRequester.DecisionStep = gridIndex % DECISION_PERIOD;
                    }

                    RacerView view = instance.AddComponent<RacerView>();
                    view.Initialize(racerId, _race, position.z, agent);
                    instance.AddComponent<DustTrailView>();
                    instance.AddComponent<CreatureAudioView>();

                    _spawned.Add(instance);
                    _racerRoots.Add(instance.transform);
                    racers.Add(new RacerState
                    {
                        RacerId = racerId,
                        CreatureId = entry.id,
                        DisplayName = $"{entry.displayName} {gridIndex + 1}",
                        Status = RacerStatus.Racing
                    });
                    gridIndex++;
                }
            }

            _cameraDirector.SetTargets(_racerRoots);
            if (racers.Count == 0)
            {
                Debug.LogWarning("No valid racers could be spawned; returning to menu.");
                RequestMenu();
                return;
            }
            _race.StartRace(racers);
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
