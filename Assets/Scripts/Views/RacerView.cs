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
    /// </summary>
    public sealed class RacerView : MonoBehaviour
    {
        private const float KNOCKDOWN_SECONDS = 5f;
        private const float KNOCKDOWN_SPEED = 0.1f;
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
                _race.NotifyFinish(_racerId);
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
                _race.NotifyFailure(_racerId);
                enabled = false;
                gameObject.SetActive(false);
                return;
            }

            _race.ReportProgress(_racerId, z - _startZ);
        }
    }
}
