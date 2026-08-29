using UnityEngine;

using PoRacer.IsaacPorts;

namespace IsaacH1
{
    /// <summary>Adapts a plain Transform to <see cref="ITargetProvider"/>.</summary>
    [DisallowMultipleComponent]
    public class IsaacH1TransformTarget : MonoBehaviour, ITargetProvider
    {
        public Transform target;

        public bool TryGetTarget(out Vector3 worldPosition)
        {
            if (target == null) { worldPosition = default; return false; }
            worldPosition = target.position;
            return true;
        }
    }
}
