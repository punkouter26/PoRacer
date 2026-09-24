namespace PoRacer.CreatureRace
{
    public enum CreatureSelfTestKind
    {
        /// <summary>Every action zero (the rest pose): must stay still, at rest, at its height, upright.</summary>
        RestPose = 0,
        /// <summary>One joint driven to rest + a fixed angle: it must read positive and move a probe point the right way.</summary>
        JointSign = 1,
    }
}
