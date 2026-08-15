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

        private void Awake()
        {
            _creature = (ICreatureAgent)_agent;
            _creature.MaxStep = TRAINING_MAX_STEP;
            _creature.SetGoal(_goal);
            _creature.SetAreaResetCallback(ResetArea);
            if (_trackRoot != null)
            {
                _trackBuilder = new Systems_TrackBuilder(_groundMaterial, _obstacleMaterial, _physicsMaterial);
                _trackBuilder.Build(_trackKind, _trackRoot, width: 12f, length: 24f, _rng);
            }
        }

        private void ResetArea()
        {
            // Obstacle tracks reshuffle every episode so the policy generalizes.
            if (_trackBuilder != null && (_trackKind == TrackKind.Bumps || _trackKind == TrackKind.Walls))
            {
                _trackBuilder.Build(_trackKind, _trackRoot, width: 12f, length: 24f, _rng);
            }
            float distance = Random.Range(MIN_GOAL_DISTANCE, MAX_GOAL_DISTANCE);
            _goal.position = _spawnPoint.position + Vector3.forward * distance
                + Vector3.up * Systems_TrackBuilder.SurfaceHeight(_trackKind, distance);

            ArticulationBody root = _creature.Root;
            Vector3 spawnLift = Vector3.up * Systems_TrackBuilder.SurfaceHeight(_trackKind, 0f);
            root.TeleportRoot(_spawnPoint.position + spawnLift, _spawnPoint.rotation);
            ResetBody(root);
            ArticulationBody[] bodies = root.GetComponentsInChildren<ArticulationBody>();
            for (int bodyIndex = 0; bodyIndex < bodies.Length; bodyIndex++)
            {
                ResetBody(bodies[bodyIndex]);
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
