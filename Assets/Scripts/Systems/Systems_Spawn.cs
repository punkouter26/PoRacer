using System;
using System.Collections.Generic;
using Cysharp.Threading.Tasks;
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
    /// in grid rows behind the start line (marathon start). After a race the
    /// results panel stays up until the player picks RACE AGAIN (RaceAgain) or
    /// MENU (RequestMenu). BeginRacing() is called by the menu.
    /// A racer whose .onnx model is missing is skipped with a warning (never crashes).
    /// </summary>
    public sealed class Systems_Spawn : IStartable, IDisposable
    {
        private const int DECISION_PERIOD = 5;

        // Visible quirks: same brain, mild physics tweak, HUD badge. Racers are
        // named "<creature> #n" by their position within their own group, so a
        // quirk no longer contributes a name prefix.
        private readonly struct QuirkDef
        {
            public readonly string Tag;
            public readonly float Power;
            public readonly float MassScale;
            public readonly float Weight;
            public readonly Color Badge;

            public QuirkDef(string tag, float power, float massScale, float weight, Color badge)
            {
                Tag = tag;
                Power = power;
                MassScale = massScale;
                Weight = weight;
                Badge = badge;
            }
        }

        private static readonly QuirkDef[] Quirks =
        {
            new(string.Empty, 1f, 1f, 0.30f, Color.clear),
            new("MIGHTY", 1.06f, 1f, 0.16f, new Color(1f, 0.45f, 0.2f)),
            new("SLEEPY", 0.94f, 1f, 0.16f, new Color(0.4f, 0.6f, 1f)),
            new("TURBO", 1.12f, 1f, 0.10f, new Color(1f, 0.85f, 0.2f)),
            new("HEAVY", 1f, 1.12f, 0.14f, new Color(0.65f, 0.65f, 0.7f)),
            new("FEATHER", 1f, 0.9f, 0.14f, new Color(0.95f, 0.95f, 0.85f))
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
        private static readonly int MetallicId = Shader.PropertyToID("_Metallic");
        private static readonly int SpecColorId = Shader.PropertyToID("_SpecColor");
        private static readonly int SurfaceIdId = Shader.PropertyToID("_SurfaceId");

        // SH_Creature pattern ids. The shader generates its own detail, so the
        // roster gets visibly different skins without a single texture asset:
        // segmented bodies read as scaled, the arthropods as plated, and the
        // humanoids as woven cloth.
        private const float SURFACE_SPECKLE = 0f;
        private const float SURFACE_SCALES = 1f;
        private const float SURFACE_PLATES = 2f;
        private const float SURFACE_WEAVE = 3f;
        private const int GRID_COLUMNS = 10;
        private const float GRID_X_SPACING = 2f;
        private const float GRID_ROW_SPACING = 1.6f;
        // Big fields start as a tower: keep a small footprint and stack layers
        // upward, so the start is a glorious collapsing pile.
        private const int STACK_THRESHOLD = 30;
        private const int STACK_FOOTPRINT_ROWS = 3;
        // How far the ground must reach behind the grid, independent of how many
        // rows the grid needs. The pack camera sits 0.8x its framing distance
        // behind the pack, and the bottom of a 9:16 frame from there lands about
        // 0.42x that distance behind the pack centre — roughly 19 m at the 45 m
        // the framing maths hits for a full-width grid. At the old 7 m the lower
        // quarter of a portrait screen was empty black past the ground's edge.
        private const float CAMERA_BACKDROP_MARGIN = 24f;
        private const float STACK_LAYER_HEIGHT = 1.5f;
        // Course starts: rows step this far along the road from its first knot,
        // columns sit this far apart across it (as many as the road's width holds,
        // which on the Acrobat road is one - single file down the centreline), and
        // the whole line starts a little past the knot so nobody is born on the
        // start gate's footing. Generous on purpose: every racer needs clear road
        // ahead and behind to get going, and the course is long enough to spend it.
        private const float COURSE_ROW_SPACING = 2.5f;
        private const float COURSE_COLUMN_SPACING = 1.4f;
        private const float COURSE_START_OFFSET = 1.5f;
        private const float STACK_JITTER = 0.35f;

        private readonly CreatureCatalog _catalog;
        private readonly RaceConfigModel _config;
        private readonly RaceTrackView _track;
        private readonly Systems_Race _race;
        private readonly Systems_CameraDirector _cameraDirector;
        private readonly RaceModel _raceModel;
        private readonly Systems_TrackBuilder _trackBuilder;
        private readonly Systems_AudioMix _audioMix;
        private readonly System.Random _rng = new();
        private CancellationTokenSource _cts = new();
        private readonly List<GameObject> _spawned = new();
        private readonly List<Transform> _racerRoots = new();
        private readonly MaterialPropertyBlock _tintBlock = new();
        private bool _racingLoopActive;
        // Incremented on every BeginRacing/RequestMenu; async chains capture it and
        // bail out after each await if a newer session has superseded them.
        private int _generation;
        private TrackKind _currentTrack = TrackKind.Flat;
        // The authored course being raced, or null on builder maps.
        private RaceCourseView _course;

        public Systems_Spawn(
            CreatureCatalog catalog,
            RaceConfigModel config,
            RaceTrackView track,
            Systems_Race race,
            Systems_CameraDirector cameraDirector,
            RaceModel raceModel,
            Systems_TrackBuilder trackBuilder,
            Systems_AudioMix audioMix)
        {
            _catalog = catalog;
            _config = config;
            _track = track;
            _race = race;
            _cameraDirector = cameraDirector;
            _raceModel = raceModel;
            _trackBuilder = trackBuilder;
            _audioMix = audioMix;
        }

        public void Start()
        {
            // Every raceable creature enters with 1 racer by default; the menu
            // shows those counts pre-selected on first launch.
            for (int entryIndex = 0; entryIndex < _catalog.Entries.Count; entryIndex++)
            {
                CreatureCatalog.CreatureEntry entry = _catalog.Entries[entryIndex];
                if (entry.prefab != null && entry.HasBrain)
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
            RunRaceGuarded(++_generation, _cts.Token).Forget();
        }

        /// <summary>Results-panel button: same roster and map, fresh grid.</summary>
        public void RaceAgain()
        {
            if (_config.MenuVisible)
            {
                return;
            }
            _racingLoopActive = true;
            RunRestartGuarded(++_generation, _cts.Token).Forget();
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
            _cts.Cancel();
            _cts.Dispose();
        }

        private bool IsCurrent(int generation) => _racingLoopActive && generation == _generation;

        /// <summary>
        /// The scene's baked geometry for exactly this map, or null. Length and
        /// features are part of the test, not just the kind: Flat, Gale and
        /// Roulette are all TrackKind.Flat and differ only by length and hazards,
        /// so matching on kind alone would race the wrong track on two of them.
        /// Roulette re-rolls its features every race and so never matches.
        /// </summary>
        private RaceTrackView.AuthoredTrack FindAuthoredTrack(
            Systems_MapCatalog.MapEntry map, TrackFeatures rolledFeatures)
        {
            IReadOnlyList<RaceTrackView.AuthoredTrack> authored = _track.AuthoredTracks;
            for (int authoredIndex = 0; authoredIndex < authored.Count; authoredIndex++)
            {
                RaceTrackView.AuthoredTrack candidate = authored[authoredIndex];
                if (candidate.root == null || candidate.kind != _currentTrack)
                {
                    continue;
                }
                // A course carries its own length and has no builder features;
                // there is one per kind, so the kind is the whole match.
                if (candidate.course != null)
                {
                    return candidate;
                }
                if (candidate.features == rolledFeatures
                    && Mathf.Approximately(candidate.lengthMeters, map.LengthMeters))
                {
                    return candidate;
                }
            }
            return null;
        }

        /// <summary>Shows one authored subtree and hides the rest; null hides them all.</summary>
        private void ShowAuthoredTrack(RaceTrackView.AuthoredTrack shown)
        {
            IReadOnlyList<RaceTrackView.AuthoredTrack> authored = _track.AuthoredTracks;
            for (int authoredIndex = 0; authoredIndex < authored.Count; authoredIndex++)
            {
                Transform root = authored[authoredIndex].root;
                if (root != null)
                {
                    root.gameObject.SetActive(authored[authoredIndex] == shown);
                }
            }
        }

        private static void ClearChildren(Transform parent)
        {
            if (parent == null)
            {
                return;
            }
            for (int childIndex = parent.childCount - 1; childIndex >= 0; childIndex--)
            {
                UnityEngine.Object.Destroy(parent.GetChild(childIndex).gameObject);
            }
        }

        /// <summary>
        /// Crash guard around the whole spawn chain: a track-builder or prefab
        /// exception must never strand the game with no racers and no menu.
        /// </summary>
        private async UniTaskVoid RunRaceGuarded(int generation, CancellationToken token)
        {
            try
            {
                await SpawnAndStartRace(generation, token);
            }
            catch (OperationCanceledException)
            {
            }
            catch (Exception exception)
            {
                Debug.LogError($"Race spawn failed; returning to menu. {exception}");
                if (IsCurrent(generation))
                {
                    RequestMenu();
                }
            }
        }

        private async UniTaskVoid RunRestartGuarded(int generation, CancellationToken token)
        {
            try
            {
                Despawn();
                // The old roster is history. Clearing it keeps the HUD from
                // listing last race's results over the new grid.
                _raceModel.ClearRacers();
                // Destroy() is deferred to end of frame; spawning immediately would
                // overlap new worms with old colliders and explode the physics.
                await UniTask.NextFrame(token);
                if (!IsCurrent(generation))
                {
                    return;
                }
                await SpawnAndStartRace(generation, token);
            }
            catch (OperationCanceledException)
            {
            }
            catch (Exception exception)
            {
                Debug.LogError($"Race spawn failed; returning to menu. {exception}");
                if (IsCurrent(generation))
                {
                    RequestMenu();
                }
            }
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
                float backMargin = Mathf.Max(CAMERA_BACKDROP_MARGIN, gridRows * GRID_ROW_SPACING + 4f);

                // A map whose geometry is baked into the scene is used as authored:
                // the runtime builder is skipped entirely, so what is tuned in the
                // Scene view is what races. Everything else is still generated.
                // The match is on kind, length and features together, because three
                // maps share TrackKind.Flat and differ only in those two.
                RaceTrackView.AuthoredTrack authoredTrack = FindAuthoredTrack(map, rolledFeatures);
                _course = authoredTrack != null ? authoredTrack.course : null;
                if (_currentTrack == TrackKind.Course && _course == null)
                {
                    // The builder cannot make a course; without the scene entry the
                    // race would run on nothing. Fall back to the first map, loudly.
                    Debug.LogError($"Map '{map.DisplayName}' needs an authored course in the scene " +
                        "(run Editor_BuildCourseTrack.Build()); racing Flat instead.");
                    map = Systems_MapCatalog.Get(0);
                    _currentTrack = map.Kind;
                    _raceModel.TrackName = map.DisplayName;
                    authoredTrack = FindAuthoredTrack(map, rolledFeatures);
                }
                ShowAuthoredTrack(authoredTrack);
                // The scene's z-line finish gate has no meaning on a course, which
                // carries its own finish trigger; hide it there and restore it after.
                if (_track.FinishLine != null)
                {
                    _track.FinishLine.gameObject.SetActive(_course == null);
                }
                if (_course != null)
                {
                    CourseFinishView courseFinish = _course.GetComponentInChildren<CourseFinishView>(true);
                    if (courseFinish != null)
                    {
                        courseFinish.Initialize(_race);
                    }
                }
                if (authoredTrack != null)
                {
                    ClearChildren(_track.TrackRoot);
                    _trackBuilder.UseAuthored(
                        authoredTrack.groundBounds, authoredTrack.hasArch, authoredTrack.archBounds);
                }
                else
                {
                    _trackBuilder.Build(_currentTrack, _track.TrackRoot, width: 24f, length: map.LengthMeters, _rng,
                        decorate: true, finishZ: finishZ, features: rolledFeatures, backMargin: backMargin);
                }

                // The finish arch has no collider, so the camera cannot discover
                // it by sweeping; hand its volume over as an explicit keep-out.
                if (_trackBuilder.TryGetFinishArchBounds(out Bounds archBounds))
                {
                    _cameraDirector.SetKeepOut(archBounds);
                }
                else
                {
                    _cameraDirector.ClearKeepOut();
                }
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
            Vector3 gridOrigin = _track.SpawnPoints.Count > 0
                ? _track.SpawnPoints[0].parent.position
                : Vector3.zero;
            Systems_CoursePath coursePath = _course != null ? _course.Path : null;
            _cameraDirector.SetCourse(coursePath);
            if (coursePath != null)
            {
                gridOrigin = coursePath.Start;
                _raceModel.TrackLengthMeters = Mathf.Max(1f, coursePath.Length);
            }
            else if (_track.FinishLine != null)
            {
                _raceModel.TrackLengthMeters = Mathf.Max(1f, _track.FinishLine.position.z - gridOrigin.z);
            }
            // Bounds every racer's runaway guard. Falling back to a wide box
            // matters: a zero-size Bounds would put every racer out of bounds on
            // its first frame and rescue the whole grid in place, forever.
            if (!_trackBuilder.TryGetGroundBounds(out Bounds groundBounds))
            {
                groundBounds = new Bounds(gridOrigin, new Vector3(120f, 1f, 120f));
            }

            // The MuJoCo world has to stand before the first Fido is instantiated, not
            // after: every MjComponent's OnEnable reads MjScene.Instance, and that getter
            // creates an MjScene when none exists. Build ours second and MjScene.Awake
            // throws "singleton, yet multiple instances found". Its model compiles in
            // Start, at the end of this frame, which for any field small enough to skip
            // the per-25 yield below is after every racer is placed; larger fields land
            // their stragglers through the plug-in's own scene-recreation request.
            if (RosterNeedsMujoco())
            {
                Systems_MujocoWorld.Build();
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
                if (entry.prefab == null || !entry.HasBrain)
                {
                    Debug.LogWarning($"Creature '{entry.id}' has no trained brain yet; skipping its {requested} racers.");
                    continue;
                }
                if (coursePath != null && entry.prefab.GetComponentInChildren<IMujocoCreature>(true) != null)
                {
                    // MuJoCo steps its own world and cannot see Unity colliders: on
                    // a course it would fall straight through the road to y = 0.
                    Debug.LogWarning($"Creature '{entry.id}' is simulated by MuJoCo and cannot run an authored course; skipping.");
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
                    Vector3 position;
                    Quaternion rotation = entry.prefab.transform.rotation;
                    if (coursePath != null)
                    {
                        // The foot of the course: rows step along the road from its
                        // first knot, columns fan across it as far as the road is
                        // wide, and each racer is set down on the surface the probe
                        // finds under it - never at a guessed height. Every racer
                        // faces down the road.
                        int courseColumns = Mathf.Max(1, Mathf.FloorToInt(_course.HalfWidth * 2f / COURSE_COLUMN_SPACING));
                        int courseRow = gridIndex / courseColumns;
                        int courseColumn = gridIndex % courseColumns;
                        float along = COURSE_START_OFFSET + courseRow * COURSE_ROW_SPACING;
                        float lateral = (courseColumn - (courseColumns - 1) * 0.5f) * COURSE_COLUMN_SPACING;
                        Vector3 heading = coursePath.HeadingAt(along);
                        Vector3 across = Vector3.Cross(Vector3.up, heading);
                        Vector3 centre = coursePath.PointAt(along);
                        Vector3 surface = centre;
                        // Over the verge the probe finds the hillside instead of the
                        // road; pull in toward the centreline until it finds the road.
                        for (int attempt = 0; attempt < 3; attempt++)
                        {
                            if (_course.TrySurfaceAt(centre + across * lateral, centre.y, out surface))
                            {
                                break;
                            }
                            lateral *= 0.5f;
                            surface = centre;
                        }
                        position = surface + Vector3.up * (entry.spawnHeight + 0.05f + layer * STACK_LAYER_HEIGHT);
                        rotation = Quaternion.LookRotation(heading, Vector3.up) * rotation;
                    }
                    else
                    {
                        position = gridOrigin + new Vector3(
                            localX,
                            Systems_TrackBuilder.SurfaceHeight(_currentTrack, localX, localZ) + entry.spawnHeight + 0.05f
                                + layer * STACK_LAYER_HEIGHT,
                            localZ);
                    }

                    // The prefab's own rotation, not identity: the Centipede is
                    // authored lying down (90 deg on X) and spawns as a
                    // collapsing vertical tower of capsules without it.
                    GameObject instance = UnityEngine.Object.Instantiate(entry.prefab, position, rotation);
                    // Generation-scoped so an orphan from a superseded spawn chain can
                    // never collide with a live racer's ID in RaceModel.
                    string racerId = $"{entry.id}#{generation}.{gridIndex + 1}";
                    instance.name = racerId;

                    // Any ICreatureAgent races: ML-Agents creatures carry BehaviorParameters,
                    // Inference-Engine creatures (IsaacBox, IsaacH1) drive their own policy.
                    ICreatureAgent agent = FindCreatureAgent(instance);
                    BehaviorParameters behavior = instance.GetComponentInChildren<BehaviorParameters>();
                    if (agent == null || (agent is Unity.MLAgents.Agent && behavior == null))
                    {
                        Debug.LogWarning($"Creature '{entry.id}' prefab lacks an ICreatureAgent; skipping.");
                        UnityEngine.Object.Destroy(instance);
                        continue;
                    }
                    if (behavior != null)
                    {
                        if (entry.model == null)
                        {
                            // No .onnx to load, so fall back to the coded gait rather
                            // than spawning a racer with no controller at all. Kept
                            // after the scripted-brains toggle was removed: it is the
                            // guard against a catalog entry losing its model.
                            behavior.BehaviorType = BehaviorType.HeuristicOnly;
                        }
                        else
                        {
                            behavior.Model = entry.model;
                            behavior.BehaviorType = BehaviorType.InferenceOnly;
                            behavior.InferenceDevice = InferenceDevice.Burst;
                        }
                    }
                    agent.MaxStep = 0;
                    CourseGoalView courseGoal = null;
                    if (coursePath != null)
                    {
                        // The carrot: a private goal kept ahead of this racer along
                        // the centreline, so the brain's goal direction follows the
                        // road. Despawned with the racers.
                        var goalObject = new GameObject(racerId + ".goal");
                        courseGoal = goalObject.AddComponent<CourseGoalView>();
                        courseGoal.Initialize(coursePath, agent.Body);
                        _spawned.Add(goalObject);
                        agent.SetGoal(goalObject.transform);
                        if (agent is Agent_Creature creatureAgent)
                        {
                            creatureAgent.SetCourse(courseGoal);
                        }
                    }
                    else
                    {
                        agent.SetGoal(_track.FinishLine);
                    }

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

                    bool isMujoco = instance.GetComponent<IMujocoCreature>() != null;

                    // Fido is bare: BodyLinkView draws links between
                    // ArticulationBodies, and he has none to link.
                    bool bareBody = isMujoco;
                    // Connective limb visuals must exist before tinting so the
                    // links pick up this racer's color along with its parts.
                    if (!bareBody)
                    {
                        instance.AddComponent<BodyLinkView>();
                    }

                    // Unique tint per racer via property block: shared material
                    // stays shared, so batching is not broken by material clones.
                    // Alternating light/dark segments plus a touch of gloss give
                    // the primitive bodies visible form under the sun.
                    // A racer that arrived with its own skin, clothes and shoes keeps
                    // them; the per-slot hue exists to tell primitive bodies apart.
                    bool authoredAppearance = instance.GetComponent<IAuthoredAppearance>() != null;
                    Color tint = Color.HSVToRGB(gridIndex * TINT_HUE_STEP % 1f, 0.6f, 1f);
                    Color darkTint = new Color(tint.r * 0.78f, tint.g * 0.78f, tint.b * 0.78f);
                    float surfaceId = SurfaceIdFor(entry.id);
                    Renderer[] tintRenderers = authoredAppearance
                        ? System.Array.Empty<Renderer>()
                        : instance.GetComponentsInChildren<Renderer>();
                    for (int rendererIndex = 0; rendererIndex < tintRenderers.Length; rendererIndex++)
                    {
                        Color segmentTint = rendererIndex % 2 == 0 ? tint : darkTint;
                        Color specTint = Color.Lerp(segmentTint, Color.white, 0.45f);
                        _tintBlock.Clear();
                        _tintBlock.SetColor(BaseColorId, segmentTint);
                        _tintBlock.SetColor(LegacyColorId, segmentTint);
                        _tintBlock.SetColor(SpecColorId, specTint);
                        _tintBlock.SetFloat(SmoothnessId, 0.72f);
                        _tintBlock.SetFloat(MetallicId, 0.22f);
                        _tintBlock.SetFloat(SurfaceIdId, surfaceId);
                        tintRenderers[rendererIndex].SetPropertyBlock(_tintBlock);
                    }

                    // Every body-tracking view and the camera must follow the transform the
                    // creature actually moves, which the agent names via Body. For most of
                    // the catalog that is the prefab root and nothing changes. Two are not:
                    // IsaacH1's prefab root is a container with the articulation starting at
                    // its `pelvis` child, and Fido is stepped by MuJoCo, which moves his
                    // torso and leaves the imported container at the start line. Keyed off
                    // `instance`, either would race perfectly while the camera framed an
                    // empty grid slot and progress, standings, flip check and dust trail all
                    // read grid row 0.
                    GameObject creatureRoot = agent.Body != null ? agent.Body.gameObject : instance;

                    RacerView view = creatureRoot.AddComponent<RacerView>();
                    float finishZ = _track.FinishLine != null ? _track.FinishLine.position.z : float.PositiveInfinity;
                    // Progress is measured from the common start line (grid row 0),
                    // not this racer's own spawn row — otherwise back-row racers
                    // report inflated progress and corrupt the leader ranking.
                    view.Initialize(racerId, _race, gridOrigin, agent, finishZ, _currentTrack, groundBounds);
                    if (courseGoal != null)
                    {
                        view.SetCourse(courseGoal, coursePath, _course.Bounds);
                    }
                    creatureRoot.AddComponent<SpeedRibbonView>().Initialize(tint);
                    creatureRoot.AddComponent<DustTrailView>();
                    // Handed the buses at spawn: the view has no scope to inject from.
                    creatureRoot.AddComponent<CreatureAudioView>().Initialize(_audioMix);
                    creatureRoot.AddComponent<SkidMarkView>().Initialize(_currentTrack);

                    _spawned.Add(instance);
                    _racerRoots.Add(creatureRoot.transform);
                    racers.Add(new RacerState
                    {
                        RacerId = racerId,
                        CreatureId = entry.id,
                        DisplayName = $"{entry.displayName} #{racerIndex + 1}",
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
            _race.StartRace(racers, map.TimeLimitSeconds);
        }

        /// <summary>
        /// Surface pattern for a creature id, consumed by SH_Creature. Unknown ids
        /// fall through to the plain speckle rather than to an arbitrary skin, so
        /// a newly added creature looks deliberate until it is classified here.
        /// </summary>
        /// <summary>
        /// True when the roster about to be spawned contains a creature MuJoCo has to
        /// step. Asked of the prefabs before any of them exists in the scene, because the
        /// MuJoCo world has to be standing before the first one is instantiated.
        /// </summary>
        private bool RosterNeedsMujoco()
        {
            for (int entryIndex = 0; entryIndex < _catalog.Entries.Count; entryIndex++)
            {
                CreatureCatalog.CreatureEntry entry = _catalog.Entries[entryIndex];
                if (entry.prefab == null || _config.GetCount(entry.id) <= 0)
                {
                    continue;
                }
                if (entry.prefab.GetComponent<IMujocoCreature>() != null)
                {
                    return true;
                }
            }
            return false;
        }

        private static float SurfaceIdFor(string creatureId)
        {
            // Catalog ids carry a version suffix ("Crab_v01"); the pattern is a
            // property of the body plan, not of the brain revision.
            int suffixIndex = creatureId.IndexOf('_');
            string bodyPlan = suffixIndex > 0 ? creatureId.Substring(0, suffixIndex) : creatureId;
            switch (bodyPlan.ToLowerInvariant())
            {
                case "centipede":
                    return SURFACE_SCALES;
                case "crab":
                case "hexapod":
                case "quad":
                case "kangaroo":
                    return SURFACE_PLATES;
                case "grandma":
                case "grandpa":
                case "matt":
                case "nick":
                case "halfbiped":
                case "biped":
                    return SURFACE_WEAVE;
                default:
                    return SURFACE_SPECKLE;
            }
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
            ICreatureAgent agent = FindCreatureAgent(instance);
            if (agent != null)
            {
                agent.NotifyDrivesChanged();
            }
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
            // After the racers, not before: MjScene.OnDestroy frees the native model that
            // any surviving CreatureAgent would still be stepping.
            Systems_MujocoWorld.Teardown();
        }
    }
}
