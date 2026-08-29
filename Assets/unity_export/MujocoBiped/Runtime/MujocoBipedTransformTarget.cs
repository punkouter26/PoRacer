using UnityEngine;

using PoRacer.IsaacPorts;

namespace MujocoBiped
{
    /// <summary>Adapts a plain Transform to <see cref="ITargetProvider"/>.</summary>
    [DisallowMultipleComponent]
    [AddComponentMenu("MujocoBiped/MujocoBiped Transform Target")]
    public class MujocoBipedTransformTarget : MonoBehaviour, ITargetProvider
    {
        [Tooltip("The object to chase. In SCN_RACE_FLAT this is FinishLine.")]
        public Transform target;

        public bool TryGetTarget(out Vector3 worldPosition)
        {
            // Unity's == is overridden to catch destroyed objects; `is null` and `?.` are
            // not, and would call through to a destroyed Transform.
            if (target == null) { worldPosition = default; return false; }
            worldPosition = target.position;
            return true;
        }
    }
}
