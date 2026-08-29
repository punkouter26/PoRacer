using UnityEngine;

namespace IsaacBiped2
{
    /// <summary>
    /// Where the biped should walk. Implemented by the ring sampler (standalone testing) and by
    /// the race adapter (which points it at the finish line).
    /// </summary>
    public interface ITargetProvider
    {
        Vector3 GetTargetWorld();
    }
}
