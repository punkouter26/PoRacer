namespace PoRacer.Agents
{
    /// <summary>
    /// Marks a racer that MuJoCo simulates rather than PhysX.
    ///
    /// Systems_Spawn needs this in two places, and both used to test for the
    /// Fido adapter by name — which silently did the wrong thing the moment a
    /// second MuJoCo racer existed. Fido was removed on 2026-09-10; the
    /// interface stays, because naming a concrete type here was the bug:
    ///
    ///   * the MuJoCo world must be built BEFORE the first such racer is
    ///     instantiated. Every MjComponent's OnEnable reads MjScene.Instance, and
    ///     that getter creates an MjScene when none exists, so building second
    ///     throws "singleton, yet multiple instances found". A roster that needed
    ///     the world but contained no Fido would never have built it at all.
    ///   * they are "bare": BodyLinkView draws links between ArticulationBodies,
    ///     and a MuJoCo racer has none to link.
    ///
    /// And a third, for RacerView: a MuJoCo racer is never switched off mid-race.
    /// Disabling any MjComponent makes the plug-in recreate the whole MuJoCo scene; on a
    /// course that recreation fails, and every MuJoCo racer in the race froze on its
    /// start line with an error per physics step. A ruled-out MuJoCo racer is held
    /// still where it lies instead (<see cref="HoldStill"/>).
    /// </summary>
    public interface IMujocoCreature
    {
        /// <summary>Stop the policy and hold the standing-stance targets: out of the race,
        /// left lying where it fell, and no MuJoCo component disabled.</summary>
        void HoldStill();
    }

    /// <summary>
    /// Marks a racer that arrives with its own authored art and must keep it.
    ///
    /// Systems_Spawn tints every racer with a per-grid-slot hue so the primitive
    /// bodies read apart at a glance. That is right for the built creatures and
    /// wrong for one wearing an authored skin, shirt and shoes — CLAUDE.md reads a
    /// creature that arrives with its own materials as a variation, not the
    /// baseline, and reserves the legend colours for other things.
    /// </summary>
    public interface IAuthoredAppearance
    {
    }
}
