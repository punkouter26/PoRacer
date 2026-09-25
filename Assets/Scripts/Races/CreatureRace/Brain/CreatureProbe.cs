using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// Race and self-test telemetry a racer's view reads off its body each physics step.
    /// None of it reaches the policy.
    /// </summary>
    public readonly struct CreatureProbe
    {
        public CreatureProbe(Vector3 lead, float referenceHeight, float referenceUpright, Vector3 probeOffset)
        {
            Lead = lead;
            ReferenceHeight = referenceHeight;
            ReferenceUpright = referenceUpright;
            ProbeOffset = probeOffset;
        }

        /// <summary>Unity world point that has to cross the finish line (the worm's nose, the quad's chest).</summary>
        public Vector3 Lead { get; }
        /// <summary>Reference body origin height above the floor, metres.</summary>
        public float ReferenceHeight { get; }
        /// <summary>Reference body up . world up: 1 upright, 0 on its side, -1 on its back.</summary>
        public float ReferenceUpright { get; }
        /// <summary>
        /// The self-test probe: point P on body B minus body A's origin, in A's frame, MuJoCo
        /// convention (x forward, y left, z up). Which A, B and P comes from the pilot.
        /// </summary>
        public Vector3 ProbeOffset { get; }
    }
}
