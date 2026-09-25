namespace PoRacer.Models
{
    /// <summary>
    /// What one racer's body and brain are doing right now, plus a short history for the
    /// telemetry card's graphs. Written by Systems_RacerTelemetry, read by the camera
    /// director (who is falling, who is getting up) and by the telemetry card.
    ///
    /// Everything is preallocated at registration: the per-frame writes never allocate.
    /// </summary>
    public sealed class RacerTelemetry
    {
        /// <summary>15 s of history at the 4 Hz the telemetry system samples it.</summary>
        public const int HISTORY_LENGTH = 60;

        /// <summary>More outputs than any racer's brain has; the Isaac H1 has 19, MojucuBoy 21.</summary>
        public const int MAX_ACTIONS = 64;

        /// <summary>Marks a quantity the racer cannot report, such as effort on an unlimited drive.</summary>
        public const float UNKNOWN = -1f;

        public RacerTelemetry(string racerId)
        {
            RacerId = racerId;
        }

        public string RacerId { get; }
        public TrainingSource TrainedBy { get; set; }

        /// <summary>Ground speed of the body, smoothed, in metres per second.</summary>
        public float SpeedMps { get; set; }

        /// <summary>1 = standing the way it spawned, 0 = on its side, below 0 = upside down.</summary>
        public float Upright { get; set; }

        /// <summary>How fast <see cref="Upright"/> is dropping, per second; 0 when steady or rising.</summary>
        public float FallRate { get; set; }

        /// <summary>How fast <see cref="Upright"/> is rising, per second; 0 when steady or falling.</summary>
        public float RiseRate { get; set; }

        public bool IsDown { get; set; }
        public float DownSeconds { get; set; }
        public bool IsGettingUp { get; set; }

        /// <summary>Unscaled time the racer last got back up by itself, or negative infinity.</summary>
        public float RecoveredAt { get; set; } = float.NegativeInfinity;

        public bool HasPower { get; set; }

        /// <summary>Mean |drive force| / force limit, 0..1, or <see cref="UNKNOWN"/>.</summary>
        public float EffortFraction { get; set; } = UNKNOWN;

        /// <summary>Mechanical power through the joints, smoothed, in watts.</summary>
        public float PowerWatts { get; set; }

        /// <summary>Joint work done since GO, in joules.</summary>
        public float EnergyJoules { get; set; }

        public float MassKg { get; set; }
        public float DistanceMeters { get; set; }

        /// <summary>
        /// Mechanical cost of transport: joint work / (weight x distance). Dimensionless,
        /// lower is more efficient. <see cref="UNKNOWN"/> until the racer has covered enough
        /// ground for the ratio to mean anything.
        /// </summary>
        public float CostOfTransport
        {
            get
            {
                const float MIN_DISTANCE_METERS = 2f;
                const float GRAVITY = 9.81f;
                if (!HasPower || MassKg <= 0f || DistanceMeters < MIN_DISTANCE_METERS)
                {
                    return UNKNOWN;
                }
                return EnergyJoules / (MassKg * GRAVITY * DistanceMeters);
            }
        }

        /// <summary>The brain's latest outputs, clamped to [-1, 1]; the first <see cref="ActionCount"/> are live.</summary>
        public float[] Actions { get; } = new float[MAX_ACTIONS];
        public int ActionCount { get; set; }

        /// <summary>Mean |change| of the outputs per decision, smoothed. 0 = calm, 2 = flipping end to end.</summary>
        public float Jitter { get; set; }

        /// <summary>
        /// <see cref="Jitter"/> at which a calm-to-twitchy readout reads fully twitchy. A
        /// walking gait sits well under 0.1; 0.5 means outputs swinging a quarter of their
        /// range every decision.
        /// </summary>
        public const float JITTER_FULL_SCALE = 0.5f;

        /// <summary>Jitter as 0 (calm) .. 1 (twitchy), for meters and tables.</summary>
        public float Twitchiness => Jitter <= 0f ? 0f : (Jitter >= JITTER_FULL_SCALE ? 1f : Jitter / JITTER_FULL_SCALE);

        public float[] SpeedHistory { get; } = new float[HISTORY_LENGTH];
        public float[] UprightHistory { get; } = new float[HISTORY_LENGTH];
        public float[] EffortHistory { get; } = new float[HISTORY_LENGTH];

        /// <summary>Samples written so far, capped at <see cref="HISTORY_LENGTH"/>.</summary>
        public int HistoryCount { get; private set; }

        /// <summary>Slot the next sample goes into; the oldest sample once the buffer is full.</summary>
        public int HistoryHead { get; private set; }

        /// <summary>Bumped on every history sample so a view can redraw only when there is something new.</summary>
        public int HistoryVersion { get; private set; }

        public void PushHistory(float speed, float upright, float effort)
        {
            SpeedHistory[HistoryHead] = speed;
            UprightHistory[HistoryHead] = upright;
            EffortHistory[HistoryHead] = effort;
            HistoryHead = (HistoryHead + 1) % HISTORY_LENGTH;
            if (HistoryCount < HISTORY_LENGTH)
            {
                HistoryCount++;
            }
            HistoryVersion++;
        }

        /// <summary>The sample <paramref name="age"/> steps back from the oldest one kept (0 = oldest).</summary>
        public float HistoryAt(float[] series, int age)
        {
            int oldest = HistoryCount < HISTORY_LENGTH ? 0 : HistoryHead;
            return series[(oldest + age) % HISTORY_LENGTH];
        }
    }
}
