using PoRacer.Systems;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Thin per-racer adapter added at spawn time: pushes race progress to
    /// Systems_Race each frame and reports physics failure (NaN, out of bounds)
    /// as DNF. A failed racer is deactivated so it cannot disturb the rest of
    /// the race. Also backstops the finish trigger: a racer fast enough to
    /// tunnel through the BoxCollider is finished by distance instead.
    /// The first two knockdowns earn a marshal rescue (set upright in place);
    /// the third is a DNF.
    /// </summary>
    public sealed class RacerView : MonoBehaviour
    {
        private const float KNOCKDOWN_SECONDS = 7f;
        private const float KNOCKDOWN_SPEED = 0.1f;
        private const int MAX_RESCUES = 2;
        private const float OUT_OF_BOUNDS_METERS = 100f;
        private const float FALL_OFF_Y = -10f;

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

        public string RacerId => _racerId;

        public void Initialize(string racerId, Systems_Race race, float startZ, Agents.ICreatureAgent agent, float finishZ)
        {
            _racerId = racerId;
            _race = race;
            _startZ = startZ;
            _agent = agent;
            _transform = transform;
            _finishDistance = finishZ - startZ;
        }

        private void Update()
        {
            if (_race == null)
            {
                return;
            }
            Vector3 position = _transform.position;
            bool corrupt = float.IsNaN(position.x) || float.IsNaN(position.y) || float.IsNaN(position.z)
                || position.y < FALL_OFF_Y
                || position.y > OUT_OF_BOUNDS_METERS
                || Mathf.Abs(position.x) > OUT_OF_BOUNDS_METERS
                || Mathf.Abs(position.z) > OUT_OF_BOUNDS_METERS;
            if (corrupt || (_agent != null && _agent.Failed))
            {
                _race.NotifyFailure(_racerId);
                if (!corrupt)
                {
                    // Corrupt positions are NaN/far away — no place to puff there.
                    FxUtil.KnockoutPuff(position);
                }
                enabled = false;
                gameObject.SetActive(false);
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
                var cosmetic = GetComponent<CosmeticPropView>();
                if (cosmetic != null && !cosmetic.IsEjected)
                {
                    cosmetic.Eject();
                }

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
                enabled = false;
                gameObject.SetActive(false);
                return;
            }

            _race.ReportProgress(_racerId, z - _startZ);
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
            // underneath it, or standing a snake "upright" means standing its
            // capsule chain on end.
            Quaternion upright = Quaternion.Euler(0f, root.transform.rotation.eulerAngles.y, 0f)
                * (_agent != null ? _agent.RestRotation : Quaternion.identity);
            root.TeleportRoot(position + Vector3.up * 0.25f, upright);
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
        }
    }
}
