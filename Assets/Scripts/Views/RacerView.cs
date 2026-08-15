using PoRacer.Systems;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Thin per-racer adapter added at spawn time: pushes race progress to
    /// Systems_Race each frame and reports physics failure (NaN) as DNF.
    /// A failed racer is deactivated so it cannot disturb the rest of the race.
    /// </summary>
    public sealed class RacerView : MonoBehaviour
    {
        private const float KNOCKDOWN_SECONDS = 5f;
        private const float KNOCKDOWN_SPEED = 0.1f;

        private string _racerId;
        private Systems_Race _race;
        private float _startZ;
        private Agents.ICreatureAgent _agent;
        private Transform _transform;
        private float _flippedSeconds;
        private float _lastZ;

        public string RacerId => _racerId;

        public void Initialize(string racerId, Systems_Race race, float startZ, Agents.ICreatureAgent agent)
        {
            _racerId = racerId;
            _race = race;
            _startZ = startZ;
            _agent = agent;
            _transform = transform;
        }

        private void Update()
        {
            if (_race == null)
            {
                return;
            }
            float z = _transform.position.z;
            if (float.IsNaN(z) || (_agent != null && _agent.Failed))
            {
                _race.NotifyFailure(_racerId);
                enabled = false;
                gameObject.SetActive(false);
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
                return;
            }

            _race.ReportProgress(_racerId, z - _startZ);
        }
    }
}
