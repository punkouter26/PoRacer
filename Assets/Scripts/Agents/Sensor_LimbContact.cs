using UnityEngine;

namespace PoRacer.Sensors
{
    /// <summary>
    /// Ground-contact flag for one limb, attached at runtime by Agent_Creature to
    /// every actuated joint body so the policy can feel which limbs carry weight.
    /// Only static geometry counts (ground, obstacles, gates): anything carrying a
    /// rigidbody or articulation body is another limb, ours or a rival's.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class Sensor_LimbContact : MonoBehaviour
    {
        // OnCollisionStay fires during the physics step before observations are
        // collected; remembering contact for just over one 0.02 s step keeps the
        // flag steady without ever reporting stale contacts.
        private const float CONTACT_MEMORY_SECONDS = 0.03f;

        private float _lastContactTime = float.NegativeInfinity;

        public bool IsGrounded => Time.fixedTime - _lastContactTime <= CONTACT_MEMORY_SECONDS;

        private void OnCollisionStay(Collision collision)
        {
            Collider other = collision.collider;
            if (other != null && other.attachedRigidbody == null && other.attachedArticulationBody == null)
            {
                _lastContactTime = Time.fixedTime;
            }
        }
    }
}
