using PoRacer.Presentation;
using PoRacer.Systems;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Thin per-racer adapter added at spawn time: pushes race progress to
    /// Systems_Race each frame and reports physics failure (NaN, out of bounds)
    /// as DNF. A knocked-out racer is deactivated so it cannot disturb the rest
    /// of the race. Also backstops the finish trigger: a racer fast enough to
    /// tunnel through the BoxCollider is finished by distance instead.
    /// The first two knockdowns earn a marshal rescue (set upright in place);
    /// the third is a DNF.
    /// A racer whose articulation solver diverges is caught by the arena bounds
    /// and put back where it was rather than deleted: PhysX can throw a long
    /// joint chain kilometres away in a single step, and a racer that silently
    /// pops out of existence reads as a bug to anyone watching.
    /// </summary>
    public sealed class RacerView : MonoBehaviour
    {
        private const float KNOCKDOWN_SECONDS = 7f;
        private const float KNOCKDOWN_SPEED = 0.1f;
        private const int MAX_RESCUES = 2;
        // Arena bounds: the ground footprint plus a margin, so a divergence is
        // caught on the step it happens instead of after it has covered a
        // kilometre, and a racer that wanders off the edge is caught at the edge
        // rather than held up over open space. The ceiling is measured from each
        // racer's own spawn height so the tower start used for big fields — 30
        // racers per 1.5 m layer — does not read as an escape.
        private const float ARENA_EDGE_MARGIN = 3f;
        private const float ARENA_HEADROOM_Y = 25f;
        // Measured down from the track surface under the racer rather than from
        // world zero: Hills/Rough/Lumpy dip below zero legitimately, and a fixed
        // floor either false-triggers there or has to sit so deep that a sinking
        // racer is metres under the ground before anything catches it.
        private const float ARENA_DEPTH_BELOW_SURFACE = 2f;
        // A diverging rig re-diverges; the cooldown keeps one blow-up from
        // burning the whole recovery budget inside a single settling frame.
        private const int MAX_RECOVERIES = 3;
        private const float RECOVERY_COOLDOWN_SECONDS = 1f;
        private const float RECOVERY_LIFT = 0.25f;

        private string _racerId;
        private Systems_Race _race;
        private float _startZ;
        private Agents.ICreatureAgent _agent;
        private Transform _transform;
        private float _flippedSeconds;
        private float _lastZ;
        private float _finishDistance;
        private bool _finished;
        private int _rescuesUsed;
        private int _recoveriesUsed;
        private bool _retired;
        private float _nextRecoveryTime;
        private float _ceilingY;
        private Vector3 _lastGoodPosition;
        private Vector3 _gridOrigin;
        private TrackKind _track;
        private float _minX;
        private float _maxX;
        private float _minZ;
        private float _maxZ;

        public string RacerId => _racerId;

        public void Initialize(string racerId, Systems_Race race, Vector3 gridOrigin,
            Agents.ICreatureAgent agent, float finishZ, TrackKind track, Bounds groundBounds)
        {
            _racerId = racerId;
            _race = race;
            _gridOrigin = gridOrigin;
            _startZ = gridOrigin.z;
            _agent = agent;
            _track = track;
            _transform = transform;
            _finishDistance = finishZ - _startZ;
            _lastGoodPosition = _transform.position;
            _ceilingY = _lastGoodPosition.y + ARENA_HEADROOM_Y;
            _minX = groundBounds.min.x - ARENA_EDGE_MARGIN;
            _maxX = groundBounds.max.x + ARENA_EDGE_MARGIN;
            _minZ = groundBounds.min.z - ARENA_EDGE_MARGIN;
            _maxZ = groundBounds.max.z + ARENA_EDGE_MARGIN;
        }

        /// <summary>
        /// World-space height of the track surface under a world-space point.
        /// Systems_TrackBuilder authors terrain in grid-local coordinates, which
        /// is how spawn places racers, so the same offset applies here.
        /// </summary>
        private float SurfaceYAt(float worldX, float worldZ)
        {
            return _gridOrigin.y + Systems_TrackBuilder.SurfaceHeight(
                _track, worldX - _gridOrigin.x, worldZ - _gridOrigin.z);
        }

        private void Update()
        {
            if (_race == null)
            {
                return;
            }
            Vector3 position = _transform.position;
            if (!IsFinite(position) || !IsInsideArena(position)
                || (!_retired && _agent != null && _agent.Failed))
            {
                Recover();
                return;
            }
            _lastGoodPosition = position;
            if (_retired)
            {
                // Retired racers keep only the arena guard: no progress, no
                // knockdown referee, but never allowed to leave either.
                return;
            }

            float z = position.z;
            if (!_finished && z - _startZ >= _finishDistance)
            {
                // Backstop for racers that tunnel through the finish trigger;
                // NotifyFinish no-ops if the trigger already fired.
                _finished = true;
                _race.NotifyFinish(_racerId, z - _startZ - _finishDistance);
            }
            if (_finished)
            {
                _lastZ = z;
                return;
            }

            // Knockdown referee: on its back and going nowhere = knocked out.
            bool flipped = _transform.up.y < 0f && Mathf.Abs(z - _lastZ) / Time.deltaTime < KNOCKDOWN_SPEED;
            _flippedSeconds = flipped ? _flippedSeconds + Time.deltaTime : 0f;
            _lastZ = z;
            if (_flippedSeconds >= KNOCKDOWN_SECONDS)
            {
                if (_rescuesUsed < MAX_RESCUES)
                {
                    _rescuesUsed++;
                    _flippedSeconds = 0f;
                    _race.NotifyWipeout(_racerId, position, false);
                    RescueFlip(position);
                    return;
                }
                _race.NotifyWipeout(_racerId, position, true);
                _race.NotifyFailure(_racerId);
                FxUtil.KnockoutPuff(position);
            // Grit under the smoke: the puff alone reads as a vanish, the debris
            // reads as a crash.
            FxUtil.WipeoutDebris(position);
                enabled = false;
                gameObject.SetActive(false);
                return;
            }

            _race.ReportProgress(_racerId, z - _startZ);
        }

        private static bool IsFinite(Vector3 position)
        {
            // Covers NaN and the infinities a diverged reduced-coordinate solve
            // reaches on its way there.
            return float.IsFinite(position.x) && float.IsFinite(position.y) && float.IsFinite(position.z);
        }

        private bool IsInsideArena(Vector3 position)
        {
            return position.x >= _minX && position.x <= _maxX
                && position.z >= _minZ && position.z <= _maxZ
                && position.y >= SurfaceYAt(position.x, position.z) - ARENA_DEPTH_BELOW_SURFACE
                && position.y <= _ceilingY;
        }

        /// <summary>
        /// Put a runaway racer back on the track: its last in-bounds lane position,
        /// set down on the surface. The height is recomputed rather than restored,
        /// because the last in-bounds sample of a racer that sank is itself already
        /// underground — reusing it just puts the creature back under the floor.
        /// RescueFlip adds the clearance on top.
        /// </summary>
        private Vector3 ArenaSafePosition()
        {
            float x = Mathf.Clamp(_lastGoodPosition.x, _minX, _maxX);
            float z = Mathf.Clamp(_lastGoodPosition.z, _minZ, _maxZ);
            return new Vector3(x, Mathf.Min(SurfaceYAt(x, z), _ceilingY), z);
        }

        /// <summary>
        /// Solver divergence containment. Teleporting the root back and zeroing
        /// every joint's position and velocity rebuilds the articulation's
        /// reduced-coordinate state from scratch, which is what actually clears
        /// the divergence — clamping velocities does not, because the garbage is
        /// in the joint positions the link poses are derived from.
        /// </summary>
        private void Recover()
        {
            if (Time.time < _nextRecoveryTime)
            {
                return;
            }
            _nextRecoveryTime = Time.time + RECOVERY_COOLDOWN_SECONDS;

            Vector3 safe = ArenaSafePosition();
            RescueFlip(safe);
            _flippedSeconds = 0f;
            _lastZ = safe.z;
            _lastGoodPosition = safe;
            if (_retired)
            {
                // Already retired and pinned; this is the backstop catching a
                // body that was somehow shoved out anyway. Nothing more to score.
                return;
            }

            _recoveriesUsed++;
            bool retiring = _recoveriesUsed > MAX_RECOVERIES;
            _race.NotifyWipeout(_racerId, safe, retiring);
            if (retiring)
            {
                Retire();
            }
        }

        /// <summary>
        /// Out of recoveries: score it a DNF and stop anything driving it, but
        /// leave it lying on the track the way a stalled-out DNF already does.
        /// Pinning the articulation base is what makes that stick — a rig that
        /// has diverged three times will do it again, and an unpinned one simply
        /// resumes falling through the floor the moment nothing is watching.
        /// Drives are already zeroed by the rescue, so the links just go limp.
        /// </summary>
        private void Retire()
        {
            _retired = true;
            _race.NotifyFailure(_racerId);
            if (_agent is MonoBehaviour agentBehaviour)
            {
                agentBehaviour.enabled = false;
            }
            ArticulationBody[] bodies = GetComponentsInChildren<ArticulationBody>();
            for (int bodyIndex = 0; bodyIndex < bodies.Length; bodyIndex++)
            {
                if (bodies[bodyIndex].isRoot)
                {
                    bodies[bodyIndex].immovable = true;
                    break;
                }
            }
        }

        /// <summary>
        /// Marshal rescue: stand the creature upright where it lies (keeping its
        /// heading), calm every joint, and puff some dust. Same reset recipe the
        /// training areas use, so the brain resumes from a familiar pose.
        /// </summary>
        private void RescueFlip(Vector3 position)
        {
            ArticulationBody[] bodies = GetComponentsInChildren<ArticulationBody>();
            ArticulationBody root = null;
            for (int bodyIndex = 0; bodyIndex < bodies.Length; bodyIndex++)
            {
                if (bodies[bodyIndex].isRoot)
                {
                    root = bodies[bodyIndex];
                    break;
                }
            }
            if (root == null)
            {
                return;
            }
            // Heading only on the yaw; the creature's authored rest pose is kept
            // underneath it, or standing a centipede "upright" means standing its
            // capsule chain on end. A diverged rig can report a NaN rotation, and
            // a NaN yaw would teleport it straight back out of the world, so fall
            // back to facing down the track.
            float yaw = root.transform.rotation.eulerAngles.y;
            if (!float.IsFinite(yaw))
            {
                yaw = 0f;
            }
            Quaternion upright = Quaternion.Euler(0f, yaw, 0f)
                * (_agent != null ? _agent.RestRotation : Quaternion.identity);
            root.TeleportRoot(position + Vector3.up * RECOVERY_LIFT, upright);
            for (int bodyIndex = 0; bodyIndex < bodies.Length; bodyIndex++)
            {
                ArticulationBody body = bodies[bodyIndex];
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
            FxUtil.KnockoutPuff(position);
            // Grit under the smoke: the puff alone reads as a vanish, the debris
            // reads as a crash.
            FxUtil.WipeoutDebris(position);
        }
    }
}
