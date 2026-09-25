using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Collision relay attached at runtime to every solid limb collider of a
    /// creature. Unity only delivers collision messages to the GameObject that owns
    /// the collider, so the racer root cannot hear its own footfalls without one of
    /// these per limb. Reports contacts up to the racer's CreatureAudioView, which
    /// owns the rate limiting and the actual playback.
    ///
    /// Three kinds of contact, told apart here:
    ///
    ///   * Static geometry — no rigidbody, no articulation body. The ground and the
    ///     scenery: a footstep.
    ///   * Another racer's limb — a body whose relay reports a different owner.
    ///     A clash.
    ///   * This creature's own limbs — a body whose relay reports the same owner.
    ///     Dropped, or an articulated creature would rattle constantly against
    ///     itself as its own segments jostle.
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
            Collider other = collision.collider;
            if (other == null)
            {
                return;
            }
            // GetContact avoids the array allocation that collision.contacts makes.
            ContactPoint contact = collision.GetContact(0);
            float impactSpeed = collision.relativeVelocity.magnitude;

            // Ground and scenery are static: no rigidbody, no articulation body.
            if (other.attachedRigidbody == null && other.attachedArticulationBody == null)
            {
                _owner.ReportLimbImpact(impactSpeed, contact.point);
            }
            else if (IsRival(other))
            {
                _owner.ReportRivalImpact(impactSpeed, contact.point);
            }
        }

        /// <summary>
        /// True when <paramref name="other"/> belongs to a different racer.
        ///
        /// Every solid limb of every racer carries one of these relays, so the
        /// cheap answer is to read the other collider's relay and compare owners.
        /// The GetComponentInParent fallback covers a collider that never got a
        /// relay — a prop, or a limb added after the spawner's pass — and only runs
        /// when the direct lookup misses, keeping the hierarchy walk out of the
        /// common case.
        /// </summary>
        private bool IsRival(Collider other)
        {
            if (other.TryGetComponent(out LimbContactView relay))
            {
                return relay._owner != _owner;
            }
            CreatureAudioView owner = other.GetComponentInParent<CreatureAudioView>();
            // No owner at all is scenery with a body — a loose prop. Not a rival.
            return owner != null && owner != _owner;
        }
    }
}
