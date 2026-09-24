namespace PoRacer.Models
{
    /// <summary>
    /// Why a racer was scored DNF. The status alone says only that it went out; the
    /// reason says what to fix. The 2026-09-24 smoke run knocked out eleven of twelve
    /// racers on the two courses and recorded none of the causes, which left falling,
    /// stalling and going over the verge indistinguishable.
    /// </summary>
    public enum KnockoutReason
    {
        None,
        /// <summary>On its back and going nowhere for RacerView's knockdown window.</summary>
        KnockedDown,
        /// <summary>No forward progress for Systems_Race.NO_PROGRESS_TIMEOUT_SECONDS.</summary>
        Stalled,
        /// <summary>Outside the arena bounds, usually over the edge of the road.</summary>
        LeftTrack,
        /// <summary>A non-finite pose: the physics solver diverged.</summary>
        Diverged,
        /// <summary>The agent reported Failed itself (the Isaac ports' own fall check).</summary>
        AgentFailed,
        /// <summary>Still racing when the podium filled; out by rule, not by failing.</summary>
        PodiumCutoff,
        /// <summary>A caller that did not say.</summary>
        Unspecified
    }
}
