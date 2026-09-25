using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// One pooled MuJoCo stand-in for a piece of fruit (AGENTS rule M): a mocap body with a
    /// sphere geom, built with the MuJoCo world, parked far below it until a piece borrows
    /// it. While bound it sits on the piece's centre, so MuJoCo racers bump into the fruit;
    /// when the piece is gone it parks itself again and is free for the next one.
    ///
    /// Pooled rather than created per piece because adding a MuJoCo body after the model
    /// compiles makes the plug-in recreate the whole scene, which resets every MuJoCo racer
    /// mid-race. One-way: MuJoCo treats a mocap body as immovable, so the racer is pushed
    /// by the fruit, and the fruit is pushed by the racer's PhysX stand-ins instead.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class MujocoFruitProxyView : MonoBehaviour
    {
        private Transform _self;
        private Transform _target;
        private Vector3 _localCentre;
        private Vector3 _parked;

        /// <summary>True while no fruit is bound, so the pool can hand it out.</summary>
        public bool IsFree => _target == null;

        public float Radius { get; private set; }

        public void Initialize(float radius, Vector3 parked)
        {
            _self = transform;
            Radius = radius;
            _parked = parked;
            _self.position = parked;
        }

        /// <summary>Follows <paramref name="target"/>, centred on its renderer bounds.</summary>
        public void Bind(Transform target)
        {
            _target = target;
            Renderer shape = target.GetComponentInChildren<Renderer>();
            _localCentre = shape != null ? target.InverseTransformPoint(shape.bounds.center) : Vector3.zero;
            Follow();
        }

        private void FixedUpdate()
        {
            if (_target == null)
            {
                if (_self.position != _parked)
                {
                    _self.position = _parked;
                }
                return;
            }
            Follow();
        }

        private void Follow()
        {
            _self.SetPositionAndRotation(_target.TransformPoint(_localCentre), _target.rotation);
        }
    }
}
