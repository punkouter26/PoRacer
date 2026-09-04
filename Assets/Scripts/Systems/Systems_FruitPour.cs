using System;
using System.Collections.Generic;
using System.Threading;
using Cysharp.Threading.Tasks;
using MessagePipe;
using PoRacer.Models;
using PoRacer.Views;
using UnityEngine;
using VContainer.Unity;

namespace PoRacer.Systems
{
    /// <summary>
    /// The produce shower. Every race end - podium complete, full time, or
    /// everyone knocked out - rains fruit and veg from the KIRI pack over the
    /// raced area as real rigidbodies: they bounce off the track and whoever is
    /// still lying on it, and pile up until the next race starts or the menu
    /// comes back, when they are cleared. Models are scaled to creature size
    /// on spawn (a real kiwi is 6 cm; nobody would see it).
    ///
    /// On an authored course there is a second trigger: when nobody in the
    /// field has gained ground for STALL_SECONDS, the produce is released at
    /// the top of the course with a shove downhill, and rolls the ramp toward
    /// the stalled pack. It re-arms after each release.
    /// </summary>
    public sealed class Systems_FruitPour : IStartable, ITickable, IDisposable
    {
        private const int PIECES_PER_POUR = 150;
        private const float POUR_SECONDS = 4f;
        // Smallest and largest extent a piece is scaled to, metres.
        private const float MIN_PIECE_SIZE = 0.3f;
        private const float MAX_PIECE_SIZE = 0.7f;
        private const float DROP_HEIGHT = 14f;
        private const float DROP_HEIGHT_JITTER = 6f;
        private const float DROP_SPEED = 2f;
        private const float DROP_SPIN = 4f;
        private const float PIECE_MASS = 0.4f;
        private const float PIECE_DRAG = 0.05f;
        private const float PIECE_ANGULAR_DRAG = 0.05f;
        // Rolling stock: the scans keep their real shape (a convex hull each, so
        // a pumpkin tumbles and a carrot skids) but sit on a slick material,
        // because at default friction a hull stops dead on an 8% grade.
        private const float PIECE_FRICTION = 0.05f;
        private const float PIECE_BOUNCE = 0.25f;
        // Inside the ground's edges, so the pile lands on the track and not off it.
        private const float EDGE_INSET = 1.5f;
        private const string ROOT_NAME = "FruitPour";
        // Course stall: the field's best course progress has not grown by
        // STALL_MIN_PROGRESS in STALL_SECONDS of racing.
        private const float STALL_SECONDS = 10f;
        private const float STALL_MIN_PROGRESS = 0.05f;
        // The release point: TOP_SPAN metres of road either side of the course's
        // summit (its highest knot - not the finish, which sits below a final
        // descent), set down just above the surface and pushed downhill.
        private const float TOP_SPAN = 8f;
        private const float GRADIENT_PROBE = 2f;
        private const float ROLL_RELEASE_HEIGHT = 0.6f;
        private const float ROLL_SPEED = 3f;
        // Frame-rate guard: the oldest pieces go when a fresh release would exceed this.
        private const int MAX_LIVE_PIECES = 300;
        // Anything this far under the lowest ground has left the world; there is no
        // floor under the mountain, and a piece that tunnels through the road
        // would otherwise fall for the rest of the race.
        private const float KILL_DEPTH = 8f;
        private const int KILL_SWEEP_FRAMES = 15;

        private readonly FruitCatalog _catalog;
        private readonly RaceTrackView _track;
        private readonly Systems_TrackBuilder _trackBuilder;
        private readonly Systems_Spawn _spawn;
        private readonly RaceConfigModel _config;
        private readonly RaceModel _raceModel;
        private readonly ISubscriber<RaceFinishedMessage> _finished;
        private readonly ISubscriber<RaceStartedMessage> _started;
        private readonly List<GameObject> _live = new(PIECES_PER_POUR);
        private readonly System.Random _rng = new();
        private IDisposable _subscriptions;
        private CancellationTokenSource _pourCts;
        private Transform _root;
        private PhysicsMaterial _rollingMaterial;
        private bool _watchingStall;
        private int _sweepCountdown;
        private float _bestProgress;
        private float _lastProgressTime;

        public Systems_FruitPour(
            FruitCatalog catalog,
            RaceTrackView track,
            Systems_TrackBuilder trackBuilder,
            Systems_Spawn spawn,
            RaceConfigModel config,
            RaceModel raceModel,
            ISubscriber<RaceFinishedMessage> finished,
            ISubscriber<RaceStartedMessage> started)
        {
            _catalog = catalog;
            _track = track;
            _trackBuilder = trackBuilder;
            _spawn = spawn;
            _config = config;
            _raceModel = raceModel;
            _finished = finished;
            _started = started;
        }

        public void Start()
        {
            var bag = DisposableBag.CreateBuilder();
            _finished.Subscribe(_ => { _watchingStall = false; BeginPour(); }).AddTo(bag);
            _started.Subscribe(_ => { Clear(); ArmStallWatch(); }).AddTo(bag);
            _subscriptions = bag.Build();
            _config.Changed += OnConfigChanged;
        }

        public void Dispose()
        {
            _config.Changed -= OnConfigChanged;
            _subscriptions?.Dispose();
            Clear();
        }

        private void OnConfigChanged()
        {
            if (_config.MenuVisible)
            {
                _watchingStall = false;
                Clear();
            }
        }

        /// <summary>
        /// Course races only: the stall watch reads the field's best course
        /// progress from the race model every tick and releases from the top
        /// when it has not moved for STALL_SECONDS.
        /// </summary>
        public void Tick()
        {
            SweepEscapees();
            if (!_watchingStall || !_raceModel.RaceActive)
            {
                return;
            }
            float best = float.NegativeInfinity;
            for (int racerIndex = 0; racerIndex < _raceModel.Racers.Count; racerIndex++)
            {
                RacerState racer = _raceModel.Racers[racerIndex];
                if (racer.Status == RacerStatus.Racing && racer.Progress > best)
                {
                    best = racer.Progress;
                }
            }
            if (best > _bestProgress + STALL_MIN_PROGRESS)
            {
                _bestProgress = best;
                _lastProgressTime = _raceModel.ElapsedSeconds;
            }
            if (_raceModel.ElapsedSeconds - _lastProgressTime >= STALL_SECONDS)
            {
                _lastProgressTime = _raceModel.ElapsedSeconds;
                BeginRoll();
            }
        }

        private void SweepEscapees()
        {
            if (_live.Count == 0 || --_sweepCountdown > 0)
            {
                return;
            }
            _sweepCountdown = KILL_SWEEP_FRAMES;
            float floorY = KillFloorY();
            for (int pieceIndex = _live.Count - 1; pieceIndex >= 0; pieceIndex--)
            {
                GameObject piece = _live[pieceIndex];
                if (piece == null)
                {
                    _live.RemoveAt(pieceIndex);
                    continue;
                }
                float y = piece.transform.position.y;
                if (!float.IsFinite(y) || y < floorY)
                {
                    UnityEngine.Object.Destroy(piece);
                    _live.RemoveAt(pieceIndex);
                }
            }
        }

        private float KillFloorY()
        {
            RaceCourseView course = _spawn.ActiveCourse;
            if (course != null)
            {
                return course.Bounds.min.y - KILL_DEPTH;
            }
            if (_trackBuilder.TryGetGroundBounds(out Bounds ground))
            {
                return ground.min.y - KILL_DEPTH;
            }
            return -KILL_DEPTH;
        }

        private void ArmStallWatch()
        {
            _watchingStall = _spawn.ActiveCourse != null && _catalog != null && _catalog.Models.Count > 0;
            _bestProgress = float.NegativeInfinity;
            _lastProgressTime = _raceModel.ElapsedSeconds;
        }

        /// <summary>The stall release: from the top of the course, rolling down. Earlier pieces stay.</summary>
        private void BeginRoll()
        {
            RaceCourseView course = _spawn.ActiveCourse;
            if (course == null)
            {
                return;
            }
            if (_pourCts != null)
            {
                _pourCts.Cancel();
                _pourCts.Dispose();
            }
            _pourCts = new CancellationTokenSource();
            RollAsync(course, _pourCts.Token).Forget();
        }

        private async UniTaskVoid RollAsync(RaceCourseView course, CancellationToken token)
        {
            EnsureRoot();
            Systems_CoursePath path = course.Path;
            float interval = POUR_SECONDS / PIECES_PER_POUR;
            float summit = SummitAlong(path);
            for (int pieceIndex = 0; pieceIndex < PIECES_PER_POUR; pieceIndex++)
            {
                float along = Mathf.Clamp(summit + Jitter(TOP_SPAN), 0f, path.Length);
                Vector3 heading = path.HeadingAt(along);
                Vector3 across = Vector3.Cross(Vector3.up, heading);
                Vector3 centre = path.PointAt(along);
                Vector3 point = centre + across * Jitter(course.HalfWidth * 0.6f);
                if (!course.TrySurfaceAt(point, centre.y, out Vector3 surface))
                {
                    surface = centre;
                }
                // Downhill along the road from here: whichever way the road drops.
                float ahead = path.PointAt(Mathf.Min(path.Length, along + GRADIENT_PROBE)).y;
                float behind = path.PointAt(Mathf.Max(0f, along - GRADIENT_PROBE)).y;
                Vector3 downhill = ahead < behind ? heading : -heading;
                Vector3 shove = downhill * ROLL_SPEED + new Vector3(Jitter(0.5f), 0f, Jitter(0.5f));
                SpawnPiece(surface + Vector3.up * ROLL_RELEASE_HEIGHT, shove);
                await UniTask.Delay(TimeSpan.FromSeconds(interval), cancellationToken: token);
            }
        }

        /// <summary>Course distance of the highest point of the road, sampled every metre.</summary>
        private static float SummitAlong(Systems_CoursePath path)
        {
            float best = 0f;
            float bestY = float.NegativeInfinity;
            for (float along = 0f; along <= path.Length; along += 1f)
            {
                float y = path.PointAt(along).y;
                if (y > bestY)
                {
                    bestY = y;
                    best = along;
                }
            }
            return best;
        }

        private void EnsureRoot()
        {
            if (_root == null)
            {
                _root = new GameObject(ROOT_NAME).transform;
            }
        }

        private void TrimToCap()
        {
            while (_live.Count >= MAX_LIVE_PIECES)
            {
                GameObject oldest = _live[0];
                _live.RemoveAt(0);
                if (oldest != null)
                {
                    UnityEngine.Object.Destroy(oldest);
                }
            }
        }

        private void BeginPour()
        {
            if (_catalog == null || _catalog.Models.Count == 0)
            {
                return;
            }
            Clear();
            _pourCts = new CancellationTokenSource();
            PourAsync(_pourCts.Token).Forget();
        }

        private void Clear()
        {
            if (_pourCts != null)
            {
                _pourCts.Cancel();
                _pourCts.Dispose();
                _pourCts = null;
            }
            for (int pieceIndex = 0; pieceIndex < _live.Count; pieceIndex++)
            {
                if (_live[pieceIndex] != null)
                {
                    UnityEngine.Object.Destroy(_live[pieceIndex]);
                }
            }
            _live.Clear();
        }

        private async UniTaskVoid PourAsync(CancellationToken token)
        {
            EnsureRoot();
            float interval = POUR_SECONDS / PIECES_PER_POUR;
            for (int pieceIndex = 0; pieceIndex < PIECES_PER_POUR; pieceIndex++)
            {
                Vector3 fall = Vector3.down * DROP_SPEED + new Vector3(Jitter(1f), 0f, Jitter(1f));
                SpawnPiece(DropPoint(), fall);
                await UniTask.Delay(TimeSpan.FromSeconds(interval), cancellationToken: token);
            }
        }

        private void SpawnPiece(Vector3 origin, Vector3 velocity)
        {
            GameObject model = _catalog.Models[_rng.Next(_catalog.Models.Count)];
            if (model == null)
            {
                return;
            }
            TrimToCap();
            Quaternion rotation = UnityEngine.Random.rotationUniform;
            GameObject piece = UnityEngine.Object.Instantiate(model, origin, rotation, _root);
            piece.name = model.name;

            // Scale to creature size off the model's own bounds, then give the
            // piece one convex hull per mesh so it tumbles and stacks.
            Renderer[] renderers = piece.GetComponentsInChildren<Renderer>();
            Bounds bounds = default;
            bool hasBounds = false;
            for (int rendererIndex = 0; rendererIndex < renderers.Length; rendererIndex++)
            {
                if (!hasBounds)
                {
                    bounds = renderers[rendererIndex].bounds;
                    hasBounds = true;
                }
                else
                {
                    bounds.Encapsulate(renderers[rendererIndex].bounds);
                }
            }
            float extent = hasBounds ? Mathf.Max(bounds.size.x, bounds.size.y, bounds.size.z) : 0.1f;
            float targetSize = Mathf.Lerp(MIN_PIECE_SIZE, MAX_PIECE_SIZE, (float)_rng.NextDouble());
            // Multiply, never replace: the FBX root already carries the file's unit
            // scale (100 - the scans are in centimetres), and the measured extent
            // includes it.
            piece.transform.localScale *= targetSize / Mathf.Max(0.001f, extent);

            if (_rollingMaterial == null)
            {
                _rollingMaterial = new PhysicsMaterial("FruitRolling")
                {
                    staticFriction = PIECE_FRICTION,
                    dynamicFriction = PIECE_FRICTION,
                    bounciness = PIECE_BOUNCE,
                    frictionCombine = PhysicsMaterialCombine.Minimum,
                    bounceCombine = PhysicsMaterialCombine.Maximum,
                };
            }
            MeshFilter[] filters = piece.GetComponentsInChildren<MeshFilter>();
            for (int filterIndex = 0; filterIndex < filters.Length; filterIndex++)
            {
                if (filters[filterIndex].sharedMesh == null)
                {
                    continue;
                }
                var hull = filters[filterIndex].gameObject.AddComponent<MeshCollider>();
                hull.sharedMesh = filters[filterIndex].sharedMesh;
                hull.convex = true;
                hull.sharedMaterial = _rollingMaterial;
            }
            var body = piece.AddComponent<Rigidbody>();
            body.mass = PIECE_MASS;
            body.linearDamping = PIECE_DRAG;
            body.angularDamping = PIECE_ANGULAR_DRAG;
            body.interpolation = RigidbodyInterpolation.None;
            // Small, fast, and released onto a road with thin kerb walls and a
            // tunnel lining: discrete detection lets a piece tunnel straight
            // through the wall it should bounce off.
            body.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
            body.linearVelocity = velocity;
            body.angularVelocity = new Vector3(Jitter(DROP_SPIN), Jitter(DROP_SPIN), Jitter(DROP_SPIN));
            _live.Add(piece);
        }

        /// <summary>
        /// Somewhere above the raced area: across the ground footprint on a
        /// builder map, along the road on an authored course.
        /// </summary>
        private Vector3 DropPoint()
        {
            float lift = DROP_HEIGHT + (float)_rng.NextDouble() * DROP_HEIGHT_JITTER;
            RaceCourseView course = _spawn.ActiveCourse;
            if (course != null)
            {
                Systems_CoursePath path = course.Path;
                float along = (float)_rng.NextDouble() * path.Length;
                Vector3 across = Vector3.Cross(Vector3.up, path.HeadingAt(along));
                return path.PointAt(along) + across * Jitter(course.HalfWidth) + Vector3.up * lift;
            }
            if (!_trackBuilder.TryGetGroundBounds(out Bounds ground))
            {
                Vector3 origin = _track.TrackRoot != null ? _track.TrackRoot.position : Vector3.zero;
                ground = new Bounds(origin, new Vector3(20f, 1f, 30f));
            }
            float x = Mathf.Lerp(ground.min.x + EDGE_INSET, ground.max.x - EDGE_INSET, (float)_rng.NextDouble());
            float z = Mathf.Lerp(ground.min.z + EDGE_INSET, ground.max.z - EDGE_INSET, (float)_rng.NextDouble());
            return new Vector3(x, ground.max.y + lift, z);
        }

        private float Jitter(float amplitude) => ((float)_rng.NextDouble() * 2f - 1f) * amplitude;
    }
}
