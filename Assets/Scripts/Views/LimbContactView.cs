using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Collision relay attached at runtime to every solid limb collider of a
    /// creature. Unity only delivers collision messages to the GameObject that owns
    /// the collider, so the racer root cannot hear its own footfalls without one of
    /// these per limb. Reports ground contacts up to the racer's CreatureAudioView,
    /// which owns the rate limiting and the actual playback.
    ///
    /// Contacts against anything carrying a body — other limbs, other racers — are
    /// dropped here, so self-collision inside an articulated creature never reaches
    /// the audio path.
    /// </summary>
    [DisallowMultipleComponent]
    internal sealed class LimbContactView : MonoBehaviour
    {
        private CreatureAudioView _owner;

        internal void Bind(CreatureAudioView owner)
        {
            _owner = owner;
        }

        private void OnCollisionEnter(Collision collision)
        {
            if (_owner == null || collision.contactCount == 0)
            {
                return;
            }
            // Ground and scenery are static: no rigidbody, no articulation body.
            // Anything else is a limb (ours or a rival's) and stays silent.
            Collider other = collision.collider;
            if (other == null || other.attachedRigidbody != null || other.attachedArticulationBody != null)
            {
                return;
            }
            // GetContact avoids the array allocation that collision.contacts makes.
            ContactPoint contact = collision.GetContact(0);
            _owner.ReportLimbImpact(collision.relativeVelocity.magnitude, contact.point);
        }
    }
}
