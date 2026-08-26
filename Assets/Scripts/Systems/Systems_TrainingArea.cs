using PoRacer.Agents;
using UnityEngine;

namespace PoRacer.Systems
{
    /// <summary>
    /// Training-scene-only referee for one self-contained area: resets the
    /// creature articulation and randomizes the task per the trainer's curriculum
    /// knobs — goal distance and angle, terrain kind and roughness, race hazards
    /// (mud, boost pads, gusts, gates), and race-day quirks (joint power, body
    /// mass, ground friction). This component must never exist in a race scene
    /// (race prefabs keep MaxStep 0 and are reset by despawn/respawn instead).
    /// </summary>
    public sealed class Systems_TrainingArea : MonoBehaviour
    {
        private const float MIN_GOAL_DISTANCE = 3f;
        private const float MAX_GOAL_DISTANCE = 20f;
        private const int TRAINING_MAX_STEP = 3000; // 60 s at 0.02 s per step
        private const float TRACK_WIDTH = 12f;
        private const float TRACK_LENGTH = 24f;
        // Off-center goals must stay on the built ground with a safety margin.
        private const float GOAL_X_MARGIN = 1f;
        private const float MAX_GOAL_ANGLE_DEGREES = 45f;

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
        private TrackFeatures _activeFeatures = TrackFeatures.None;
        // Authored values, cached once so per-episode quirk scaling never
        // compounds across resets.
        private ArticulationBody[] _bodies;
        private float[] _baseStiffness;
        private float[] _baseForceLimit;
        private float[] _baseMass;
        // Per-area friction material so per-episode friction quirks never touch
        // the shared asset (or other areas).
        private PhysicsMaterial _areaPhysicsMaterial;
        private float _baseStaticFriction;
        private float _baseDynamicFriction;

        private void Awake()
        {
            _creature = (ICreatureAgent)_agent;
            _creature.MaxStep = TRAINING_MAX_STEP;
            _creature.SetGoal(_goal);
            _creature.SetAreaResetCallback(ResetArea);
            _activeKind = _trackKind;

            _areaPhysicsMaterial = _physicsMaterial != null
                ? Instantiate(_physicsMaterial)
                : new PhysicsMaterial("TrainingGround");
            _baseStaticFriction = _areaPhysicsMaterial.staticFriction;
            _baseDynamicFriction = _areaPhysicsMaterial.dynamicFriction;

            if (_trackRoot != null)
            {
                _trackBuilder = new Systems_TrackBuilder(_groundMaterial, _obstacleMaterial, _areaPhysicsMaterial);
                _trackBuilder.Build(_trackKind, _trackRoot, TRACK_WIDTH, TRACK_LENGTH, _rng);
            }

            _bodies = _creature.Root.GetComponentsInChildren<ArticulationBody>();
            _baseStiffness = new float[_bodies.Length];
            _baseForceLimit = new float[_bodies.Length];
            _baseMass = new float[_bodies.Length];
            for (int bodyIndex = 0; bodyIndex < _bodies.Length; bodyIndex++)
            {
                ArticulationDrive drive = _bodies[bodyIndex].xDrive;
                _baseStiffness[bodyIndex] = drive.stiffness;
                _baseForceLimit[bodyIndex] = drive.forceLimit;
                _baseMass[bodyIndex] = _bodies[bodyIndex].mass;
            }
        }

        private void ResetArea()
        {
            // Curriculum knobs from the trainer (defaults keep gameplay behavior):
            // goal_distance_max/goal_angle_span grow the task, rough_amplitude
            // grows the terrain, track_kind swaps the map, hazard_level layers in
            // the race hazards, and the quirk_* spans randomize joint power, body
            // mass, and ground friction the way race day does.
            float maxGoalDistance = MAX_GOAL_DISTANCE;
            float goalAngleSpan = 0f;
            float quirkPowerSpan = 0f;
            float quirkMassSpan = 0f;
            float quirkFrictionSpan = 0f;
            int hazardLevel = 0;
            TrackKind requestedKind = _trackKind;
            if (Unity.MLAgents.Academy.IsInitialized)
            {
                var envParams = Unity.MLAgents.Academy.Instance.EnvironmentParameters;
                maxGoalDistance = envParams.GetWithDefault("goal_distance_max", MAX_GOAL_DISTANCE);
                goalAngleSpan = Mathf.Clamp(envParams.GetWithDefault("goal_angle_span", 0f), 0f, MAX_GOAL_ANGLE_DEGREES);
                // Clamped: above 1.0 terrain dips would cross the agents' fixed fall line (y < -1).
                Systems_TrackBuilder.RoughAmplitudeScale = Mathf.Clamp(envParams.GetWithDefault("rough_amplitude", 1f), 0f, 1f);
                quirkPowerSpan = Mathf.Clamp(envParams.GetWithDefault("quirk_power_span", 0f), 0f, 0.5f);
                quirkMassSpan = Mathf.Clamp(envParams.GetWithDefault("quirk_mass_span", 0f), 0f, 0.3f);
                quirkFrictionSpan = Mathf.Clamp(envParams.GetWithDefault("quirk_friction_span", 0f), 0f, 0.5f);
                hazardLevel = Mathf.Clamp(Mathf.RoundToInt(envParams.GetWithDefault("hazard_level", 0f)), 0, 2);
                int kindValue = Mathf.RoundToInt(envParams.GetWithDefault("track_kind", (float)_trackKind));
                if (System.Enum.IsDefined(typeof(TrackKind), kindValue))
                {
                    requestedKind = (TrackKind)kindValue;
                }
            }

            // Race hazards layered onto the training track: level 1 adds mud pits
            // and boost pads, level 2 adds gusts and gates on top. Same builder
            // path race day uses, so triggers and forces match exactly.
            TrackFeatures features = TrackFeatures.None;
            if (hazardLevel >= 1)
            {
                features |= TrackFeatures.MudPits | TrackFeatures.BoostPads;
            }
            if (hazardLevel >= 2)
            {
                features |= TrackFeatures.Gusts | TrackFeatures.Gates;
            }

            // Rebuild every episode when anything random is on the track: obstacle
            // and rough kinds reshuffle so the policy generalizes, hazards respawn
            // in new spots, and a curriculum switch also forces a rebuild.
            bool kindChanged = requestedKind != _activeKind;
            bool featuresChanged = features != _activeFeatures;
            _activeKind = requestedKind;
            _activeFeatures = features;
            bool randomizedKind = _activeKind != TrackKind.Flat && _activeKind != TrackKind.Hills;
            if (_trackBuilder != null
                && (kindChanged || featuresChanged || randomizedKind || features != TrackFeatures.None))
            {
                _trackBuilder.Build(_activeKind, _trackRoot, TRACK_WIDTH, TRACK_LENGTH, _rng, features: features);
            }

            if (quirkFrictionSpan > 0f || _areaPhysicsMaterial.staticFriction != _baseStaticFriction)
            {
                float frictionFactor = 1f + Random.Range(-quirkFrictionSpan, quirkFrictionSpan);
                _areaPhysicsMaterial.staticFriction = _baseStaticFriction * frictionFactor;
                _areaPhysicsMaterial.dynamicFriction = _baseDynamicFriction * frictionFactor;
            }

            // Goal at a random distance and heading so the policy learns to steer,
            // not just to run straight; the sideways reach is clamped onto the
            // built ground.
            float distance = Random.Range(MIN_GOAL_DISTANCE, maxGoalDistance);
            float angle = Random.Range(-goalAngleSpan, goalAngleSpan);
            Vector3 offset = Quaternion.Euler(0f, angle, 0f) * Vector3.forward * distance;
            offset.x = Mathf.Clamp(offset.x, -(TRACK_WIDTH * 0.5f - GOAL_X_MARGIN), TRACK_WIDTH * 0.5f - GOAL_X_MARGIN);
            _goal.position = _spawnPoint.position + offset
                + Vector3.up * Systems_TrackBuilder.SurfaceHeight(_activeKind, offset.x, offset.z);

            ArticulationBody root = _creature.Root;
            Vector3 spawnLift = Vector3.up * Systems_TrackBuilder.SurfaceHeight(_activeKind, 0f);
            // The spawn point only carries a heading; the creature's authored
            // rest pose has to survive the reset or a lying-down rig (snake,
            // centipede) is stood upright and explodes on the next step.
            root.TeleportRoot(_spawnPoint.position + spawnLift,
                _spawnPoint.rotation * _creature.RestRotation);
            ApplyQuirks(quirkPowerSpan, quirkMassSpan);
            ResetBody(root);
            for (int bodyIndex = 0; bodyIndex < _bodies.Length; bodyIndex++)
            {
                ResetBody(_bodies[bodyIndex]);
            }
        }

        /// <summary>
        /// Race spawning scales every racer's joint power and body mass by random
        /// factors; with non-zero spans the policy trains against that same
        /// distribution instead of meeting it for the first time on race day.
        /// Always rewrites from the authored caches so factors never compound.
        /// </summary>
        private void ApplyQuirks(float powerSpan, float massSpan)
        {
            float powerFactor = powerSpan > 0f ? 1f + Random.Range(-powerSpan, powerSpan) : 1f;
            float massFactor = massSpan > 0f ? 1f + Random.Range(-massSpan, massSpan) : 1f;
            for (int bodyIndex = 0; bodyIndex < _bodies.Length; bodyIndex++)
            {
                _bodies[bodyIndex].mass = _baseMass[bodyIndex] * massFactor;
                ArticulationDrive drive = _bodies[bodyIndex].xDrive;
                drive.stiffness = _baseStiffness[bodyIndex] * powerFactor;
                drive.forceLimit = float.IsFinite(_baseForceLimit[bodyIndex] * powerFactor)
                    ? _baseForceLimit[bodyIndex] * powerFactor
                    : _baseForceLimit[bodyIndex];
                _bodies[bodyIndex].xDrive = drive;
            }
            // Fatigue must re-capture its full-power baseline from the fresh quirks.
            _creature.NotifyDrivesChanged();
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
