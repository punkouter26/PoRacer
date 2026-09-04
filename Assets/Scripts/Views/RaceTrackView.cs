using System;
using System.Collections.Generic;
using PoRacer.Systems;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Scene anchor for the race track: lane spawn points, the finish line, and
    /// the authored tracks baked into the scene. Pure data exposure — no logic.
    /// Which authored track (if any) matches the selected map is decided by
    /// Systems_Spawn, which owns that call.
    /// </summary>
    public sealed class RaceTrackView : MonoBehaviour
    {
        /// <summary>
        /// One map's geometry, baked into the scene as ordinary GameObjects under
        /// <see cref="root"/>. Kind, length and features together identify the map
        /// it stands in for; the bounds are its footprint, read instead of measuring
        /// the hierarchy every race. A track with a <see cref="course"/> is an
        /// authored course (a GLB with its own centreline and start markers) rather
        /// than a builder map, and races along that centreline.
        /// </summary>
        [Serializable]
        public sealed class AuthoredTrack
        {
            public Transform root;
            public TrackKind kind;
            public float lengthMeters;
            public TrackFeatures features;
            public Bounds groundBounds;
            public bool hasArch;
            public Bounds archBounds;
            public RaceCourseView course;
        }

        [SerializeField] private Transform[] _spawnPoints;
        [SerializeField] private Transform _finishLine;
        [SerializeField] private Transform _trackRoot;
        [SerializeField] private Material _groundMaterial;
        [SerializeField] private Material _obstacleMaterial;
        [SerializeField] private PhysicsMaterial _physicsMaterial;

        [Header("Authored tracks")]
        [Tooltip("Track geometry baked into the scene, one entry per deterministic map. "
            + "When the selected map matches an entry's kind, length and features, "
            + "Systems_Spawn enables that subtree and skips the runtime builder entirely, "
            + "so it can be tuned by hand here and what you see in the Scene view is what "
            + "races. Roulette re-rolls every race and is always generated. Rebuild the "
            + "builder-map entries with Editor_BakeAuthoredTrack.Bake() (that discards hand "
            + "edits); course entries come from Editor_BuildCourseTrack.")]
        [SerializeField] private AuthoredTrack[] _authoredTracks = Array.Empty<AuthoredTrack>();

        public IReadOnlyList<Transform> SpawnPoints => _spawnPoints;

        public Transform FinishLine => _finishLine;

        public Transform TrackRoot => _trackRoot;

        public Material GroundMaterial => _groundMaterial;

        public Material ObstacleMaterial => _obstacleMaterial;

        public PhysicsMaterial PhysicsMaterial => _physicsMaterial;

        public IReadOnlyList<AuthoredTrack> AuthoredTracks => _authoredTracks;
    }
}
