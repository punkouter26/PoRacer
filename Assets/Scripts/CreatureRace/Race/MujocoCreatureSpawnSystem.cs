using System;
using System.Collections.Generic;
using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// Puts a MuJoCo-only creature's racers on the start line and takes them away again.
    /// Owns the parsed rig, the shared render meshes and every GameObject a race creates at
    /// runtime (the racers and the MuJoCo world); the track, camera, light and HUD are
    /// authored in the scene (AGENTS rule G) and never touched here.
    ///
    /// ONE MuJoCo world per race. Every racer is created in the same frame under the same
    /// MjScene, so the plug-in compiles them into one model: they collide with each other
    /// natively (AGENTS rule M) and each racer's view resolves its own ids and addresses.
    /// Each racer stands with its lead point <see cref="CreatureRaceConfig.StartGap"/> behind
    /// the start line, in its rest pose, at the rig's spawn height - the trainer's reset minus
    /// its yaw and joint noise.
    ///
    /// A racer set to PhysX is failed with a clear reason: this creature has no PhysX builder.
    /// </summary>
    public sealed class MujocoCreatureSpawnSystem : ICreatureSpawner, IDisposable
    {
        private readonly CreatureRaceConfig _config;
        private readonly List<GameObject> _racers = new();
        private readonly CreatureMeshCache _meshes = new();
        private CreatureLayout _layout;
        private string _error = string.Empty;
        private GameObject _mujocoWorld;

        public MujocoCreatureSpawnSystem(CreatureRaceConfig config)
        {
            _config = config;
        }

        public bool MujocoSupported => MujocoCreatureWorld.IsSupported;

        public bool TryPrepare(out CreatureLayout layout, out string error)
        {
            if (_layout == null && string.IsNullOrEmpty(_error))
            {
                _error = Prepare(out _layout);
            }
            layout = _layout;
            error = _error;
            return _layout != null;
        }

        public void Spawn(CreatureLayout layout, CreaturePilot[] pilots)
        {
            Despawn();
            IReadOnlyList<CreatureRacerDefinition> racers = _config.Racers;
            Vector3 forward = Vector3.forward;
            Vector3 setBack = forward * (layout.LeadOffset.x + _config.StartGap);

            if (MujocoSupported)
            {
                _mujocoWorld = MujocoCreatureWorld.Build(_config.MujocoSolverIterations, _config.MjcfDumpFile,
                                                         layout.Rig.FloorContact);
            }
            for (int lane = 0; lane < racers.Count && lane < pilots.Length; lane++)
            {
                CreatureRacerDefinition racer = racers[lane];
                if (racer.Physics != CreaturePhysicsKind.MujocoPlugin)
                {
                    pilots[lane].Fail($"{racer.Name}: this creature races on the MuJoCo plug-in only; set its "
                                    + "Physics to Mujoco Plugin in the settings.");
                    continue;
                }
                if (_mujocoWorld == null)
                {
                    continue;
                }
                Vector3 rootOrigin = _config.LaneOrigin(lane) - setBack;
                MujocoCreatureInstance instance = MujocoCreatureBuilder.Build(
                    layout, pilots[lane], $"{layout.Rig.Name}_L{lane}_{racer.Name}", rootOrigin, forward,
                    racer.Material, _meshes);
                _racers.Add(instance.Root);
            }
        }

        public void Despawn()
        {
            // Stop MuJoCo stepping before anything it simulates is queued for destruction.
            MujocoCreatureWorld.Suspend(_mujocoWorld);
            for (int index = 0; index < _racers.Count; index++)
            {
                if (_racers[index] != null)
                {
                    UnityEngine.Object.Destroy(_racers[index]);
                }
            }
            _racers.Clear();
            if (_mujocoWorld != null)
            {
                UnityEngine.Object.Destroy(_mujocoWorld);
            }
            _mujocoWorld = null;
        }

        public void Dispose()
        {
            Despawn();
            _meshes.Dispose();
        }

        private string Prepare(out CreatureLayout layout)
        {
            layout = null;
            TextAsset rigJson = _config.RigJson;
            if (rigJson == null)
            {
                return $"no rig JSON assigned in the settings. Copy the trainer's rig JSON under Assets/ and re-run "
                     + $"{_config.RebuildCommand}.";
            }
            if (!CreatureRigParser.TryParse(rigJson.text, out CreatureRig rig, out string error))
            {
                return $"{rigJson.name}: {error}";
            }
            return CreatureLayout.TryCreate(rig, _config.Observation, _config.LeadBody, 0f, out layout, out error)
                ? string.Empty
                : $"{rigJson.name}: {error}";
        }
    }
}
