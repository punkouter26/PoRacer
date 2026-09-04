using PoRacer.Systems;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Finish trigger of an authored course. Same job as FinishLineView, but the
    /// volume is authored in the GLB rather than being the scene's z-line, and
    /// there is one per course, so Systems_Spawn hands it the race system at
    /// race start instead of it being injected.
    /// </summary>
    [RequireComponent(typeof(BoxCollider))]
    public sealed class CourseFinishView : MonoBehaviour
    {
        private Systems_Race _race;

        public void Initialize(Systems_Race race)
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
                _race.NotifyFinish(racer.RacerId);
            }
        }
    }
}
