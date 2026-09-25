namespace PoRacer.Agents
{
    /// <summary>
    /// Joint effort for a racer PhysX does not simulate. The telemetry system reads a
    /// PhysX racer's ArticulationBody drives directly; a MuJoCo racer has no
    /// ArticulationBody, so its controller measures the same quantities from mjData and
    /// reports them through this.
    /// </summary>
    public interface IEffortReadout
    {
        /// <summary>False until the controller has bound to the compiled model.</summary>
        bool HasEffort { get; }

        /// <summary>Sum over actuators of |force x velocity|, in watts, at the last physics step.</summary>
        float MechanicalPowerWatts { get; }

        /// <summary>Mean |force| / force limit over the actuators, 0..1.</summary>
        float EffortFraction { get; }

        /// <summary>Total mass of the racer's bodies, in kilograms.</summary>
        float TotalMassKg { get; }
    }
}
