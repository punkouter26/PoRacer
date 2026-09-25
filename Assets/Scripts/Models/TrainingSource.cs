namespace PoRacer.Models
{
    /// <summary>
    /// What trained a racer's brain — the team it races for in the Sim Wars league.
    /// Values are serialized into CreatureCatalog.asset by number, so append only.
    /// </summary>
    public enum TrainingSource
    {
        Unknown = 0,
        MuJoCo = 1,
        IsaacLab = 2,
        // Unity ML-Agents (PhysX). No new training happens here (AGENTS.md rule J); the
        // brains already racing keep racing until they are retrained in MuJoCo or Isaac Lab.
        MlAgents = 3,
        // A hand-coded bot. Set at spawn, not in the catalog: an entry that loses its
        // .onnx falls back to its coded gait and races for this team instead.
        Heuristic = 4
    }
}
