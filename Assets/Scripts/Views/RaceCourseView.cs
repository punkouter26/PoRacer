using System.Collections.Generic;
using PoRacer.Systems;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Scene data for an authored course such as the Acrobat track: the
    /// centreline the racers follow, where they start, and the footprint the
    /// runaway guard uses. Written once by Editor_BuildCourseTrack from the
    /// markers inside the GLB; read by Systems_Spawn and the course training
    /// area. Pure data exposure — no logic beyond building the path object.
    /// </summary>
    public sealed class RaceCourseView : MonoBehaviour
    {
        [Tooltip("Centreline knots in world space, start to finish.")]
        [SerializeField] private Vector3[] _points = System.Array.Empty<Vector3>();
        [Tooltip("Authored start markers. Racers beyond their count fan out along the centreline behind them.")]
        [SerializeField] private Transform[] _spawnPoints = System.Array.Empty<Transform>();
        [Tooltip("World-space bounds of the drivable geometry; the runaway guard reads these.")]
        [SerializeField] private Bounds _bounds;
        [Tooltip("Half-width of the racing surface, used to keep the spawn fan on the road.")]
        [SerializeField] private float _halfWidth = 3f;

        private const float SURFACE_PROBE_UP = 3f;
        private const float SURFACE_PROBE_RANGE = 10f;
        // Further below the centreline than this and the probe has found the
        // hillside past the verge, not the road.
        private const float SURFACE_MAX_DROP = 1.5f;

        private Systems_CoursePath _path;
        private readonly RaycastHit[] _probeHits = new RaycastHit[16];

        public Systems_CoursePath Path => _path ??= new Systems_CoursePath(_points);

        public IReadOnlyList<Transform> SpawnPoints => _spawnPoints;

        public Bounds Bounds => _bounds;

        public float HalfWidth => _halfWidth;

        /// <summary>
        /// The road surface under a point, found by probing straight down onto
        /// this course's own colliders (racers and triggers are ignored). False
        /// when nothing is there or the only thing there is the hillside below
        /// the verge, so callers can pull a spawn back toward the centreline.
        /// </summary>
        public bool TrySurfaceAt(Vector3 point, float centrelineY, out Vector3 surface)
        {
            surface = point;
            int count = Physics.RaycastNonAlloc(point + Vector3.up * SURFACE_PROBE_UP, Vector3.down, _probeHits,
                SURFACE_PROBE_RANGE, ~0, QueryTriggerInteraction.Ignore);
            bool found = false;
            float bestY = float.MinValue;
            for (int hitIndex = 0; hitIndex < count; hitIndex++)
            {
                RaycastHit hit = _probeHits[hitIndex];
                if (!hit.collider.transform.IsChildOf(transform))
                {
                    continue;
                }
                if (hit.point.y > bestY)
                {
                    bestY = hit.point.y;
                    surface = hit.point;
                    found = true;
                }
            }
            return found && centrelineY - bestY <= SURFACE_MAX_DROP;
        }
    }
}
