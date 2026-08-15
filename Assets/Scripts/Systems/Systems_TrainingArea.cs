using PoRacer.Agents;
using UnityEngine;

namespace PoRacer.Systems
{
    /// <summary>
    /// Training-scene-only referee for one self-contained area: resets the worm
    /// articulation and randomizes the goal distance (3-20 m natural curriculum).
    /// This component must never exist in a race scene (race prefabs keep MaxStep 0
    /// and are reset by despawn/respawn instead).
    /// </summary>
    public sealed class Systems_TrainingArea : MonoBehaviour
    {
        private const float MIN_GOAL_DISTANCE = 3f;
        private const float MAX_GOAL_DISTANCE = 20f;
        private const int TRAINING_MAX_STEP = 3000; // 60 s at 0.02 s per step

        [SerializeField] private Unity.MLAgents.Agent _agent;
        [SerializeField] private Transform _goal;
        [SerializeField] private Transform _spawnPoint;
        [SerializeField] private TrackKind _trackKind = TrackKind.Flat;
        [SerializeField] private Transform _trackRoot;
        [SerializeField] private Material _groundMaterial;
        [SerializeField] private Material _obstacleMaterial;
        [SerializeField] private PhysicsMaterial _physicsMaterial;

        private ICreatureAgent _creature;
        private Systems_TrackBuilder _trackBuilder;
        private readonly System.Random _rng = new();
        private TrackKind _activeKind;
        // Authored drive values, cached once so per-episode quirk scaling never
        // compounds across resets.
        private ArticulationBody[] _bodies;
        private float[] _baseStiffness;
        private float[] _baseForceLimit;

        private void Awake()
        {
            _creature = (ICreatureAgent)_agent;
            _creature.MaxStep = TRAINING_MAX_STEP;
            _creature.SetGoal(_goal);
            _creature.SetAreaResetCallback(ResetArea);
            _activeKind = _trackKind;
            if (_trackRoot != null)
            {
                _trackBuilder = new Systems_TrackBuilder(_groundMaterial, _obstacleMaterial, _physicsMaterial);
                _trackBuilder.Build(_trackKind, _trackRoot, width: 12f, length: 24f, _rng);
            }

            _bodies = _creature.Root.GetComponentsInChildren<ArticulationBody>();
            _baseStiffness = new float[_bodies.Length];
            _baseForceLimit = new float[_bodies.Length];
            for (int bodyIndex = 0; bodyIndex < _bodies.Length; bodyIndex++)
            {
                ArticulationDrive drive = _bodies[bodyIndex].xDrive;
                _baseStiffness[bodyIndex] = drive.stiffness;
                _baseForceLimit[bodyIndex] = drive.forceLimit;
            }
        }

        private void ResetArea()
        {
            // Curriculum knobs from the trainer (defaults keep gameplay behavior):
            // goal_distance_max grows the task, rough_amplitude grows the terrain,
            // track_kind swaps the map, quirk_power_span randomizes joint power the
            // way race-day spawning does.
            float maxGoalDistance = MAX_GOAL_DISTANCE;
            float quirkSpan = 0f;
            TrackKind requestedKind = _trackKind;
            if (Unity.MLAgents.Academy.IsInitialized)
            {
                var envParams = Unity.MLAgents.Academy.Instance.EnvironmentParameters;
                maxGoalDistance = envParams.GetWithDefault("goal_distance_max", MAX_GOAL_DISTANCE);
                Systems_TrackBuilder.RoughAmplitudeScale = envParams.GetWithDefault("rough_amplitude", 1f);
                quirkSpan = Mathf.Clamp(envParams.GetWithDefault("quirk_power_span", 0f), 0f, 0.5f);
                int kindValue = Mathf.RoundToInt(envParams.GetWithDefault("track_kind", (float)_trackKind));
                if (System.Enum.IsDefined(typeof(TrackKind), kindValue))
                {
                    requestedKind = (TrackKind)kindValue;
                }
            }
            // Obstacle and rough tracks rebuild every episode: obstacles reshuffle so
            // the policy generalizes; a curriculum kind switch also forces a rebuild.
            bool kindChanged = requestedKind != _activeKind;
            _activeKind = requestedKind;
            if (_trackBuilder != null
                && (kindChanged || (_activeKind != TrackKind.Flat && _activeKind != TrackKind.Hills)))
            {
                _trackBuilder.Build(_activeKind, _trackRoot, width: 12f, length: 24f, _rng);
            }
            float distance = Random.Range(MIN_GOAL_DISTANCE, maxGoalDistance);
            _goal.position = _spawnPoint.position + Vector3.forward * distance
                + Vector3.up * Systems_TrackBuilder.SurfaceHeight(_activeKind, distance);

            ArticulationBody root = _creature.Root;
            Vector3 spawnLift = Vector3.up * Systems_TrackBuilder.SurfaceHeight(_activeKind, 0f);
            root.TeleportRoot(_spawnPoint.position + spawnLift, _spawnPoint.rotation);
            ApplyPowerQuirk(quirkSpan);
            ResetBody(root);
            for (int bodyIndex = 0; bodyIndex < _bodies.Length; bodyIndex++)
            {
                ResetBody(_bodies[bodyIndex]);
            }
        }

        /// <summary>
        /// Race spawning scales every racer's joint power by a random factor; with a
        /// non-zero span the policy trains against that same distribution instead of
        /// meeting it for the first time on race day.
        /// </summary>
        private void ApplyPowerQuirk(float span)
        {
            float factor = span > 0f ? 1f + Random.Range(-span, span) : 1f;
            for (int bodyIndex = 0; bodyIndex < _bodies.Length; bodyIndex++)
            {
                ArticulationDrive drive = _bodies[bodyIndex].xDrive;
                drive.stiffness = _baseStiffness[bodyIndex] * factor;
                drive.forceLimit = float.IsFinite(_baseForceLimit[bodyIndex])
                    ? _baseForceLimit[bodyIndex] * factor
                    : _baseForceLimit[bodyIndex];
                _bodies[bodyIndex].xDrive = drive;
            }
        }

        private static void ResetBody(ArticulationBody body)
        {
            if (body.jointPosition.dofCount > 0)
            {
                body.jointPosition = new ArticulationReducedSpace(0f);
                body.jointVelocity = new ArticulationReducedSpace(0f);
                ArticulationDrive drive = body.xDrive;
                drive.target = 0f;
                body.xDrive = drive;
            }
            body.linearVelocity = Vector3.zero;
            body.angularVelocity = Vector3.zero;
        }
    }
}
