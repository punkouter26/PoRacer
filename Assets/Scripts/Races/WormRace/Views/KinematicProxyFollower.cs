using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// A kinematic PhysX capsule that shadows one MuJoCo worm segment, so the PhysX worm
    /// collides with the MuJoCo worm (AGENTS rule M) even though PhysX cannot see MuJoCo.
    /// Moved with MovePosition/MoveRotation, which gives it a velocity for the contact
    /// solver instead of teleporting. It is one-way by construction: it pushes the PhysX
    /// worm and is never pushed back (MuJoCo owns the real segment).
    /// </summary>
    [RequireComponent(typeof(Rigidbody))]
    [DisallowMultipleComponent]
    public sealed class KinematicProxyFollower : MonoBehaviour
    {
        private Rigidbody _body;
        private Transform _target;

        internal void Bind(Transform target)
        {
            _target = target;
            _body = GetComponent<Rigidbody>();
        }

        private void FixedUpdate()
        {
            if (_target == null || _body == null)
            {
                return;
            }
            _body.MovePosition(_target.position);
            _body.MoveRotation(_target.rotation);
        }
    }
}
