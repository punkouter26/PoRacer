using System.Collections.Generic;

namespace PoRacer.Models
{
    /// <summary>
    /// Live telemetry for every racer on the grid, and which one the viewer has picked.
    /// Owned by Systems_RacerTelemetry; the focus is set by Systems_CameraDirector, because
    /// picking a racer is a camera action first.
    /// </summary>
    public sealed class RaceTelemetryModel
    {
        private readonly List<RacerTelemetry> _racers = new();
        private readonly Dictionary<string, RacerTelemetry> _racersById = new();

        public IReadOnlyList<RacerTelemetry> Racers => _racers;

        /// <summary>The racer the viewer picked; the telemetry card follows it. Null = no card.</summary>
        public string FocusedRacerId { get; set; }

        public RacerTelemetry Add(string racerId)
        {
            var telemetry = new RacerTelemetry(racerId);
            _racers.Add(telemetry);
            _racersById[racerId] = telemetry;
            return telemetry;
        }

        public RacerTelemetry Find(string racerId)
        {
            if (racerId == null)
            {
                return null;
            }
            return _racersById.TryGetValue(racerId, out RacerTelemetry telemetry) ? telemetry : null;
        }

        public void Clear()
        {
            _racers.Clear();
            _racersById.Clear();
            FocusedRacerId = null;
        }
    }
}
