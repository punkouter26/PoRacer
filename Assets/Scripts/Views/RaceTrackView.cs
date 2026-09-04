using System.Collections.Generic;
using PoRacer.Systems;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Scene anchor for the race track: lane spawn points, the finish line, and
    /// the optional authored track baked into the scene. Pure data exposure — no
    /// logic. Whether the authored track matches the selected map is decided by
    /// Systems_Spawn, which owns that call.
    /// </summary>
    public sealed class RaceTrackView : MonoBehaviour
    {
        [SerializeField] private Transform[] _spawnPoints;
        [SerializeField] private Transform _finishLine;
        [SerializeField] private Transform _trackRoot;
        [SerializeField] private Material _groundMaterial;
        [SerializeField] private Material _obstacleMaterial;
        [SerializeField] private PhysicsMaterial _physicsMaterial;

        [Header("Authored track")]
        [Tooltip("Track geometry baked into the scene. When the selected map matches "
            + "the kind, length and features below, Systems_Spawn enables this subtree "
            + "and skips the runtime builder entirely — so it can be tuned by hand here "
            + "and what you see in the Scene view is what races. Leave empty to always "
            + "generate. Rebuild with Editor_BakeAuthoredTrack.Bake().")]
        [SerializeField] private Transform _authoredTrack;
        [SerializeField] private TrackKind _authoredKind = TrackKind.Flat;
        [SerializeField] private float _authoredLengthMeters = 22f;
        [SerializeField] private TrackFeatures _authoredFeatures = TrackFeatures.None;
        [Tooltip("Footprint of the authored ground, in world space. The runaway guard "
            + "and the camera read these instead of measuring the baked hierarchy every race.")]
        [SerializeField] private Bounds _authoredGroundBounds;
        [SerializeField] private bool _authoredHasArch;
        [SerializeField] private Bounds _authoredArchBounds;

        public IReadOnlyList<Transform> SpawnPoints => _spawnPoints;

        public Transform FinishLine => _finishLine;

        public Transform TrackRoot => _trackRoot;

        public Material GroundMaterial => _groundMaterial;

        public Material ObstacleMaterial => _obstacleMaterial;

        public PhysicsMaterial PhysicsMaterial => _physicsMaterial;

        public Transform AuthoredTrack => _authoredTrack;

        public TrackKind AuthoredKind => _authoredKind;

        public float AuthoredLengthMeters => _authoredLengthMeters;

        public TrackFeatures AuthoredFeatures => _authoredFeatures;

        public Bounds AuthoredGroundBounds => _authoredGroundBounds;

        public bool AuthoredHasArch => _authoredHasArch;

        public Bounds AuthoredArchBounds => _authoredArchBounds;
    }
}
