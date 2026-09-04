namespace PoRacer.Agents
{
    /// <summary>
    /// Where a creature is along an authored course. The agent's goal transform
    /// still supplies the direction it observes; this supplies the distance the
    /// reward and the finish check use, so a look-ahead goal that keeps moving
    /// ahead of the racer does not read as "never getting closer".
    /// </summary>
    public interface ICourseProgress
    {
        /// <summary>Metres travelled along the course from its start line.</summary>
        float ProgressMeters { get; }

        /// <summary>Metres left to the finish line, measured along the course.</summary>
        float RemainingMeters { get; }
    }
}
