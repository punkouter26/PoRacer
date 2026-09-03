namespace PoRacer.Agents
{
    /// <summary>
    /// Marks a racer that MuJoCo simulates rather than PhysX.
    ///
    /// Systems_Spawn needs this in two places, and both used to test for
    /// <see cref="Agent_Fido"/> by name — which silently did the wrong thing the
    /// moment a second MuJoCo racer existed:
    ///
    ///   * the MuJoCo world must be built BEFORE the first such racer is
    ///     instantiated. Every MjComponent's OnEnable reads MjScene.Instance, and
    ///     that getter creates an MjScene when none exists, so building second
    ///     throws "singleton, yet multiple instances found". A roster that needed
    ///     the world but contained no Fido would never have built it at all.
    ///   * they are "bare": BodyLinkView draws links between ArticulationBodies,
    ///     and a MuJoCo racer has none to link.
    /// </summary>
    public interface IMujocoCreature
    {
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
