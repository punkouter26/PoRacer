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
    /// </summary>
    public sealed class Systems_FruitPour : IStartable, IDisposable
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
        // Inside the ground's edges, so the pile lands on the track and not off it.
        private const float EDGE_INSET = 1.5f;
        private const string ROOT_NAME = "FruitPour";

        private readonly FruitCatalog _catalog;
        private readonly RaceTrackView _track;
        private readonly Systems_TrackBuilder _trackBuilder;
        private readonly Systems_Spawn _spawn;
        private readonly RaceConfigModel _config;
        private readonly ISubscriber<RaceFinishedMessage> _finished;
        private readonly ISubscriber<RaceStartedMessage> _started;
        private readonly List<GameObject> _live = new(PIECES_PER_POUR);
        private readonly System.Random _rng = new();
        private IDisposable _subscriptions;
        private CancellationTokenSource _pourCts;
        private Transform _root;

        public Systems_FruitPour(
            FruitCatalog catalog,
            RaceTrackView track,
            Systems_TrackBuilder trackBuilder,
            Systems_Spawn spawn,
            RaceConfigModel config,
            ISubscriber<RaceFinishedMessage> finished,
            ISubscriber<RaceStartedMessage> started)
        {
            _catalog = catalog;
            _track = track;
            _trackBuilder = trackBuilder;
            _spawn = spawn;
            _config = config;
            _finished = finished;
            _started = started;
        }

        public void Start()
        {
            var bag = DisposableBag.CreateBuilder();
            _finished.Subscribe(_ => BeginPour()).AddTo(bag);
            _started.Subscribe(_ => Clear()).AddTo(bag);
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
                Clear();
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
            if (_root == null)
            {
                _root = new GameObject(ROOT_NAME).transform;
            }
            float interval = POUR_SECONDS / PIECES_PER_POUR;
            for (int pieceIndex = 0; pieceIndex < PIECES_PER_POUR; pieceIndex++)
            {
                SpawnPiece();
                await UniTask.Delay(TimeSpan.FromSeconds(interval), cancellationToken: token);
            }
        }

        private void SpawnPiece()
        {
            GameObject model = _catalog.Models[_rng.Next(_catalog.Models.Count)];
            if (model == null)
            {
                return;
            }
            Vector3 origin = DropPoint();
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

            MeshFilter[] filters = piece.GetComponentsInChildren<MeshFilter>();
            for (int filterIndex = 0; filterIndex < filters.Length; filterIndex++)
            {
                if (filters[filterIndex].sharedMesh == null)
                {
                    continue;
                }
                var collider = filters[filterIndex].gameObject.AddComponent<MeshCollider>();
                collider.sharedMesh = filters[filterIndex].sharedMesh;
                collider.convex = true;
            }
            var body = piece.AddComponent<Rigidbody>();
            body.mass = PIECE_MASS;
            body.linearDamping = PIECE_DRAG;
            body.interpolation = RigidbodyInterpolation.None;
            body.linearVelocity = Vector3.down * DROP_SPEED
                + new Vector3(Jitter(1f), 0f, Jitter(1f));
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
