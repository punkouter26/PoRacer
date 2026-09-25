using System;
using System.Collections.Generic;
using PoRacer.CreatureRace;
using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// The worm's <see cref="ICreatureSpawner"/>: puts the worms on the start line and takes
    /// them away again. Owns the parsed rig, the shared render meshes and every GameObject a
    /// race creates at runtime (the worms, the MuJoCo world, the collision stand-ins); the
    /// track, camera, light and HUD are authored in SCN_WORM_RACE and never touched here.
    ///
    /// One worm per entry of the racer list, lane = list index, each built in the physics its
    /// entry selects: MuJoCo worms through the creature template's generic
    /// <see cref="MujocoCreatureBuilder"/> (fed by <see cref="WormRigAdapter"/>), PhysX worms
    /// through the worm's own <see cref="PhysxWormBuilder"/>. Each nose starts
    /// <see cref="CreatureRaceConfig.StartGap"/> behind the start line, the worm straight and
    /// lying on the floor at the rig's spawn height.
    ///
    /// ONE MuJoCo world per race: every MuJoCo worm is created in the same frame under the
    /// same MjScene, so they collide natively and never share an index. ACROSS simulators
    /// (AGENTS rule M) every PhysX worm gets five mocap stand-ins inside the MuJoCo world and
    /// every MuJoCo worm five kinematic stand-ins in PhysX.
    ///
    /// Despawn and the next Spawn must be at least one frame apart (DOCS/Plan-P1-Worm.md D14).
    /// </summary>
    public sealed class WormSpawnSystem : ICreatureSpawner, IDisposable
    {
        private readonly WormRaceSettings _settings;
        private readonly CreatureRaceConfig _config;
        private readonly List<GameObject> _worms = new();
        private readonly List<GameObject> _kinematicProxies = new();
        private readonly List<Transform[]> _physxSegments = new();
        private readonly List<int> _physxLanes = new();
        private readonly List<Transform[]> _mujocoGeoms = new();
        private readonly List<int> _mujocoLanes = new();
        private readonly CreatureMeshCache _meshes = new();
        private WormRig _rig;
        private CreatureLayout _layout;
        private string _error = string.Empty;
        private GameObject _mujocoWorld;

        public WormSpawnSystem(WormRaceSettings settings)
        {
            _settings = settings;
            _config = settings.Race;
        }

        public bool MujocoSupported => MujocoCreatureWorld.IsSupported;

        public bool TryPrepare(out CreatureLayout layout, out string error)
        {
            if (_layout == null && string.IsNullOrEmpty(_error))
            {
                _error = Prepare();
            }
            layout = _layout;
            error = _error;
            return _layout != null;
        }

        /// <summary>
        /// Builds every racer's worm. A MuJoCo racer on a platform without MuJoCo is skipped
        /// (its pilot has already been failed by the race system).
        /// </summary>
        public void Spawn(CreatureLayout layout, CreaturePilot[] pilots)
        {
            Despawn();
            Mesh segmentMesh = _meshes.Capsule(_rig.SegmentRadius, _rig.SegmentHalfLength);

            IReadOnlyList<CreatureRacerDefinition> racers = _config.Racers;
            Vector3 forward = Vector3.forward;
            Vector3 setBack = forward * (layout.LeadOffset.x + _config.StartGap);

            // World first: the MjScene singleton must exist before any Mj component, and
            // every MuJoCo worm and stand-in must exist before it compiles (next frame).
            if (MujocoSupported && AnyRacerUses(racers, CreaturePhysicsKind.MujocoPlugin))
            {
                _mujocoWorld = MujocoCreatureWorld.Build(_config.MujocoSolverIterations, _config.MjcfDumpFile,
                                                         layout.Rig.FloorContact);
            }

            for (int lane = 0; lane < racers.Count && lane < pilots.Length; lane++)
            {
                CreatureRacerDefinition racer = racers[lane];
                Vector3 rootOrigin = _config.LaneOrigin(lane) - setBack;
                string rootName = $"Worm_L{lane}_{racer.Name}";
                if (racer.Physics == CreaturePhysicsKind.MujocoPlugin)
                {
                    if (_mujocoWorld == null)
                    {
                        continue;
                    }
                    MujocoCreatureInstance worm = MujocoCreatureBuilder.Build(
                        layout, pilots[lane], rootName, rootOrigin, forward, racer.Material, _meshes);
                    _worms.Add(worm.Root);
                    // The adapter gives the worm exactly its five segment capsules, in order.
                    _mujocoGeoms.Add(worm.Geoms);
                    _mujocoLanes.Add(lane);
                }
                else
                {
                    _worms.Add(PhysxWormBuilder.Build(_rig, _settings, layout, pilots[lane], rootName, rootOrigin,
                                                      forward, racer.Material, segmentMesh,
                                                      out Transform[] segments));
                    _physxSegments.Add(segments);
                    _physxLanes.Add(lane);
                }
            }

            if (_settings.CrossSimulatorProxies && _mujocoWorld != null
                && _mujocoGeoms.Count > 0 && _physxSegments.Count > 0)
            {
                // Same frame as the world: MuJoCo only sees what exists when it compiles.
                CreatureContact contact = WormRigAdapter.SegmentContact(_rig);
                for (int index = 0; index < _physxSegments.Count; index++)
                {
                    WormProxyBuilder.BuildMocapProxies(_mujocoWorld.transform, _physxSegments[index],
                                                       _rig, contact, _physxLanes[index]);
                }
                for (int index = 0; index < _mujocoGeoms.Count; index++)
                {
                    _kinematicProxies.Add(WormProxyBuilder.BuildKinematicProxies(
                        _mujocoGeoms[index], _rig, _settings.WormPhysicsMaterial, _mujocoLanes[index]));
                }
            }
        }

        public void Despawn()
        {
            // Stop MuJoCo stepping before anything it simulates is queued for destruction.
            MujocoCreatureWorld.Suspend(_mujocoWorld);
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
            _meshes.Dispose();
        }

        private string Prepare()
        {
            TextAsset rigJson = _config.RigJson;
            if (rigJson == null)
            {
                return $"no worm_rig.json assigned. Copy training/worm/worm_rig.json to {WormRacePaths.RIG_JSON} "
                     + "and assign it as Rig Json in the race settings asset.";
            }
            if (!WormRig.TryParse(rigJson.text, out _rig, out string error))
            {
                _rig = null;
                return error;
            }
            // The contract's float action scale, so targets and observations stay bit-identical
            // to the worm's own pre-template pilot (the rig's 0.7853981... rounds differently).
            if (!CreatureLayout.TryCreate(WormRigAdapter.ToCreatureRig(_rig), _config.Observation, _config.LeadBody,
                                          WormContract.JOINT_RANGE_RAD, out _layout, out error))
            {
                _layout = null;
                return "worm_rig.json: " + error;
            }
            return string.Empty;
        }

        private static bool AnyRacerUses(IReadOnlyList<CreatureRacerDefinition> racers, CreaturePhysicsKind physics)
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
