using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// Race and self-test telemetry a worm view reads off its body each physics step, in
    /// plain Unity world space. Kept apart from <see cref="WormBodyState"/> because none of
    /// it reaches the policy.
    /// </summary>
    internal readonly struct WormProbe
    {
        /// <summary>Tip of segment 0's capsule: the point that has to cross the line.</summary>
        public readonly Vector3 Nose;
        /// <summary>Segment 2's centre height above the floor, metres.</summary>
        public readonly float ReferenceHeight;
        /// <summary>Segment 1's centre relative to segment 0's, along the head's right.</summary>
        public readonly float SecondSegmentLateral;
        /// <summary>Segment 1's centre relative to segment 0's, along the head's up.</summary>
        public readonly float SecondSegmentVertical;

        public WormProbe(Vector3 nose, float referenceHeight, float secondSegmentLateral,
                         float secondSegmentVertical)
        {
            Nose = nose;
            ReferenceHeight = referenceHeight;
            SecondSegmentLateral = secondSegmentLateral;
            SecondSegmentVertical = secondSegmentVertical;
        }
    }
}
