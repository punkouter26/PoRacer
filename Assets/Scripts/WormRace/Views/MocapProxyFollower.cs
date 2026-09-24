using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// Keeps an MjMocapBody on one PhysX worm segment, so the MuJoCo worm collides with the
    /// PhysX worm (AGENTS rule M). The plug-in copies a mocap body's Unity transform into
    /// mjData.mocap_pos/quat after every step (MjMocapBody.OnSyncState), so moving the
    /// transform is all it takes; the contact sees it one 5 ms step late. One-way: MuJoCo
    /// treats a mocap body as infinitely heavy and never pushes the PhysX segment back.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class MocapProxyFollower : MonoBehaviour
    {
        private Transform _self;
        private Transform _target;

        internal void Bind(Transform target)
        {
            _target = target;
            _self = transform;
        }

        private void FixedUpdate()
        {
            if (_target == null)
            {
                return;
            }
            _self.SetPositionAndRotation(_target.position, _target.rotation);
        }
    }
}
