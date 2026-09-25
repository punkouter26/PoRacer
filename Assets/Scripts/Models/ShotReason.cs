namespace PoRacer.Models
{
    /// <summary>
    /// Why the broadcast director is showing what it is showing. Drives the on-screen
    /// caption; Pack, Overview and Winner carry no caption of their own.
    /// </summary>
    public enum ShotReason
    {
        Pack,
        Overview,
        Leader,
        NewLeader,
        Duel,
        GoingDown,
        GettingUp,
        BackUp,
        FinalMetres,
        ViewerPick,
        Winner
    }
}
