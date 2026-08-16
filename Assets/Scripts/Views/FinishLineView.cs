using PoRacer.Systems;
using UnityEngine;
using VContainer;

namespace PoRacer.Views
{
    /// <summary>
    /// Trigger volume across the finish line. Forwards crossings to Systems_Race.
    /// </summary>
    [RequireComponent(typeof(BoxCollider))]
    public sealed class FinishLineView : MonoBehaviour
    {
        private Systems_Race _race;

        [Inject]
        public void Construct(Systems_Race race)
        {
            _race = race;
        }

        private void OnTriggerEnter(Collider other)
        {
            if (_race == null)
            {
                return;
            }
            RacerView racer = other.GetComponentInParent<RacerView>();
            if (racer != null)
            {
                float overshoot = racer.transform.position.z - transform.position.z;
                _race.NotifyFinish(racer.RacerId, overshoot);
            }
        }
    }
}
