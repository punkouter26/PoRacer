using System.Collections.Generic;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Scene anchor for the race track: lane spawn points and the finish line.
    /// Pure data exposure — no logic.
    /// </summary>
    public sealed class RaceTrackView : MonoBehaviour
    {
        [SerializeField] private Transform[] _spawnPoints;
        [SerializeField] private Transform _finishLine;
        [SerializeField] private Transform _trackRoot;
        [SerializeField] private Material _groundMaterial;
        [SerializeField] private Material _obstacleMaterial;
        [SerializeField] private PhysicsMaterial _physicsMaterial;

        public IReadOnlyList<Transform> SpawnPoints => _spawnPoints;

        public Transform FinishLine => _finishLine;

        public Transform TrackRoot => _trackRoot;

        public Material GroundMaterial => _groundMaterial;

        public Material ObstacleMaterial => _obstacleMaterial;

        public PhysicsMaterial PhysicsMaterial => _physicsMaterial;
    }
}
