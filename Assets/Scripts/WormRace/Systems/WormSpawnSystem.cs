using System;
using System.Collections.Generic;
using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// Puts the worms on the start line and takes them away again. Owns the parsed rig, the
    /// shared segment mesh and every GameObject a race creates at runtime (the worms, the
    /// MuJoCo world, the collision stand-ins); the track, camera, light and HUD are authored
    /// in SCN_WORM_RACE and never touched here.
    ///
    /// One worm per entry of WormRaceSettings' racer list, lane = list index, each built in
    /// the physics its entry selects. Each nose starts <see cref="WormRaceSettings.StartGap"/>
    /// behind the start line, the worm straight and lying on the floor at the rig's spawn
    /// height, exactly the training reset minus its yaw and joint noise.
    ///
    /// ONE MuJoCo world per race. Every MuJoCo worm is created in the same frame, under the
    /// same MjScene, so the plug-in compiles them into one model: they collide with each
    /// other natively, and each worm's view resolves its own bodies, joints and actuators by
    /// the unique names the plug-in generates (MujocoId, QposAddress, DofAddress), so no
    /// index is ever shared between worms. Adjacent-segment excludes are per worm.
    ///
    /// ACROSS simulators (AGENTS rule M) every PhysX worm gets five mocap stand-ins inside
    /// the MuJoCo world, and every MuJoCo worm gets five kinematic stand-ins in PhysX, so
    /// every MuJoCo/PhysX pair is covered. PhysX worms collide with each other natively.
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
        private readonly List<GameObject> _worms = new();
        private readonly List<GameObject> _kinematicProxies = new();
        private readonly List<Transform[]> _physxSegments = new();
        private readonly List<int> _physxLanes = new();
        private readonly List<Transform[]> _mujocoGeoms = new();
        private readonly List<int> _mujocoLanes = new();
        private WormRig _rig;
        private string _rigError = string.Empty;
        private Mesh _segmentMesh;
        private GameObject _mujocoWorld;

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

        /// <summary>
        /// Builds every racer's worm. A MuJoCo racer on a platform without MuJoCo is skipped
        /// (its pilot has already been failed by the race system).
        /// </summary>
        internal void Spawn(WormRig rig, WormPilot[] pilots)
        {
            Despawn();
            if (_segmentMesh == null)
            {
                _segmentMesh = WormCapsuleMesh.Create(rig.SegmentRadius, rig.SegmentHalfLength);
            }

            IReadOnlyList<WormRacerDefinition> racers = _settings.Racers;
            Vector3 forward = Vector3.forward;
            Vector3 setBack = forward * (rig.NoseOffset + _settings.StartGap);
            bool mujocoAvailable = MujocoSupported;

            // World first: the MjScene singleton must exist before any Mj component, and
            // every MuJoCo worm and stand-in must exist before it compiles (next frame).
            if (mujocoAvailable && AnyRacerUses(racers, WormPhysicsKind.MujocoPlugin))
            {
                _mujocoWorld = MujocoWorldBuilder.Build(_settings.MujocoSolverIterations,
                                                        _settings.DumpMujocoMjcf, rig.Friction);
            }

            for (int lane = 0; lane < racers.Count && lane < pilots.Length; lane++)
            {
                WormRacerDefinition racer = racers[lane];
                Vector3 rootOrigin = LaneOrigin(lane) - setBack;
                string rootName = $"Worm_L{lane}_{racer.Name}";
                if (racer.Physics == WormPhysicsKind.MujocoPlugin)
                {
                    if (_mujocoWorld == null)
                    {
                        continue;
                    }
                    _worms.Add(MujocoWormBuilder.Build(rig, pilots[lane], rootName, rootOrigin, forward,
                                                       racer.Material, _segmentMesh, out Transform[] geoms));
                    _mujocoGeoms.Add(geoms);
                    _mujocoLanes.Add(lane);
                }
                else
                {
                    _worms.Add(PhysxWormBuilder.Build(rig, _settings, pilots[lane], rootName, rootOrigin, forward,
                                                      racer.Material, _segmentMesh, out Transform[] segments));
                    _physxSegments.Add(segments);
                    _physxLanes.Add(lane);
                }
            }

            if (_settings.CrossSimulatorProxies && _mujocoWorld != null
                && _mujocoGeoms.Count > 0 && _physxSegments.Count > 0)
            {
                // Same frame as the world: MuJoCo only sees what exists when it compiles.
                for (int index = 0; index < _physxSegments.Count; index++)
                {
                    WormProxyBuilder.BuildMocapProxies(_mujocoWorld.transform, _physxSegments[index],
                                                       rig, _physxLanes[index]);
                }
                for (int index = 0; index < _mujocoGeoms.Count; index++)
                {
                    _kinematicProxies.Add(WormProxyBuilder.BuildKinematicProxies(
                        _mujocoGeoms[index], rig, _settings.WormPhysicsMaterial, _mujocoLanes[index]));
                }
            }
        }

        internal void Despawn()
        {
            // Stop MuJoCo stepping before anything it simulates is queued for destruction.
            MujocoWorldBuilder.Suspend(_mujocoWorld);
            DestroyAll(_kinematicProxies);
            DestroyAll(_worms);
            DestroyIfAlive(ref _mujocoWorld);
            _physxSegments.Clear();
            _physxLanes.Clear();
            _mujocoGeoms.Clear();
            _mujocoLanes.Clear();
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

        private static bool AnyRacerUses(IReadOnlyList<WormRacerDefinition> racers, WormPhysicsKind physics)
        {
            for (int lane = 0; lane < racers.Count; lane++)
            {
                if (racers[lane].Physics == physics)
                {
                    return true;
                }
            }
            return false;
        }

        private static void DestroyAll(List<GameObject> targets)
        {
            for (int index = 0; index < targets.Count; index++)
            {
                if (targets[index] != null)
                {
                    UnityEngine.Object.Destroy(targets[index]);
                }
            }
            targets.Clear();
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
