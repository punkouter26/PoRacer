using System;
using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// Puts the two worms on the start line and takes them away again. Owns the parsed rig,
    /// the shared segment mesh and every GameObject a race creates at runtime (the worms,
    /// the MuJoCo world, the collision stand-ins); the track, camera, light and HUD are
    /// authored in SCN_WORM_RACE and never touched here.
    ///
    /// Lane 0 is the MuJoCo worm (BLUE), lane 1 the Isaac worm (ORANGE); each nose starts
    /// <see cref="WormRaceSettings.StartGap"/> behind the start line, the worm straight and
    /// lying on the floor at the rig's spawn height, exactly the training reset minus its
    /// yaw and joint noise.
    ///
    /// The builders attach each worm's physics adapter (a View) and bind it to its pilot;
    /// this system only ever holds the resulting GameObjects - the same factory boundary
    /// the race scene's Systems_Spawn has (DOCS/Plan-P1-Worm.md D13).
    ///
    /// Despawn and the next Spawn must be at least one frame apart (D14): Destroy is
    /// deferred, and MjScene's singleton only frees when the old one is really gone.
    /// </summary>
    public sealed class WormSpawnSystem : IDisposable
    {
        private readonly WormRaceSettings _settings;
        private WormRig _rig;
        private string _rigError = string.Empty;
        private Mesh _segmentMesh;
        private GameObject _mujocoWorld;
        private GameObject _mujocoWorm;
        private GameObject _isaacWorm;
        private GameObject _kinematicProxies;

        public WormSpawnSystem(WormRaceSettings settings)
        {
            _settings = settings;
        }

        internal bool MujocoSupported => MujocoWorldBuilder.IsSupported;

        internal bool TryGetRig(out WormRig rig, out string error)
        {
            if (_rig == null && string.IsNullOrEmpty(_rigError))
            {
                TextAsset rigJson = _settings.RigJson;
                if (rigJson == null)
                {
                    _rigError = $"no worm_rig.json assigned. Copy training/worm/worm_rig.json to "
                              + $"{WormRacePaths.RIG_JSON} and re-run Editor_BuildWormRaceScene.Build().";
                }
                else if (!WormRig.TryParse(rigJson.text, out _rig, out _rigError))
                {
                    _rig = null;
                }
            }
            rig = _rig;
            error = _rigError;
            return _rig != null;
        }

        /// <summary>Where a lane's worm root goes so its nose sits just behind the line.</summary>
        internal Vector3 LaneOrigin(int lane)
        {
            return new Vector3(_settings.LaneX(lane), 0f, _settings.StartLineZ);
        }

        internal void Spawn(WormRig rig, WormPilot mujocoPilot, WormPilot isaacPilot, bool spawnMujoco)
        {
            Despawn();
            if (_segmentMesh == null)
            {
                _segmentMesh = WormCapsuleMesh.Create(rig.SegmentRadius, rig.SegmentHalfLength);
            }

            Vector3 forward = Vector3.forward;
            Vector3 setBack = forward * (rig.NoseOffset + _settings.StartGap);

            Transform[] mujocoGeoms = null;
            if (spawnMujoco)
            {
                // World first: the MjScene singleton must exist before any Mj component.
                _mujocoWorld = MujocoWorldBuilder.Build(_settings.MujocoSolverIterations,
                                                        _settings.DumpMujocoMjcf, rig.Friction);
                _mujocoWorm = MujocoWormBuilder.Build(rig, _settings, mujocoPilot,
                    LaneOrigin(WormRaceModel.MUJOCO_LANE) - setBack, forward,
                    _settings.MujocoMaterial, _segmentMesh, out mujocoGeoms);
            }

            _isaacWorm = PhysxWormBuilder.Build(rig, _settings, isaacPilot,
                LaneOrigin(WormRaceModel.ISAAC_LANE) - setBack, forward,
                _settings.IsaacMaterial, _segmentMesh, out Transform[] isaacSegments);

            if (_settings.CrossSimulatorProxies && spawnMujoco)
            {
                // Same frame as the world: MuJoCo only sees what exists when it compiles.
                WormProxyBuilder.BuildMocapProxies(_mujocoWorld.transform, isaacSegments, rig);
                _kinematicProxies = WormProxyBuilder.BuildKinematicProxies(
                    mujocoGeoms, rig, _settings.WormPhysicsMaterial);
            }
        }

        internal void Despawn()
        {
            // Stop MuJoCo stepping before anything it simulates is queued for destruction.
            MujocoWorldBuilder.Suspend(_mujocoWorld);
            DestroyIfAlive(ref _kinematicProxies);
            DestroyIfAlive(ref _isaacWorm);
            DestroyIfAlive(ref _mujocoWorm);
            DestroyIfAlive(ref _mujocoWorld);
        }

        public void Dispose()
        {
            Despawn();
            if (_segmentMesh != null)
            {
                UnityEngine.Object.Destroy(_segmentMesh);
                _segmentMesh = null;
            }
        }

        private static void DestroyIfAlive(ref GameObject target)
        {
            if (target != null)
            {
                UnityEngine.Object.Destroy(target);
            }
            target = null;
        }
    }
}
