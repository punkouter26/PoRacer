using PoRacer.Models;
using UnityEngine;

namespace PoRacer.Presentation
{
    /// <summary>
    /// Names and colours of the Sim Wars teams — the tools that trained each brain.
    ///
    /// MuJoCo blue and Isaac Lab orange are the colours the Worm5 head-to-head already
    /// used, so the league reads the same as that experiment. Green is never a team colour:
    /// AGENTS.md rule D reserves it for the baseline RL racer. Red is used for one team
    /// only, the heuristic bots, because rule D already means "heuristic" by red.
    /// </summary>
    public static class TrainerTeams
    {
        public static readonly Color MuJoCo = new(0.26f, 0.56f, 1f);
        public static readonly Color IsaacLab = new(1f, 0.6f, 0.22f);
        public static readonly Color MlAgents = new(0.68f, 0.5f, 1f);
        public static readonly Color Heuristic = new(0.9f, 0.24f, 0.22f);
        public static readonly Color Unknown = new(0.6f, 0.63f, 0.68f);

        /// <summary>Teams in the order the league table lists them.</summary>
        public static readonly TrainingSource[] LeagueOrder =
        {
            TrainingSource.MuJoCo, TrainingSource.IsaacLab, TrainingSource.MlAgents, TrainingSource.Heuristic
        };

        public static string DisplayName(TrainingSource team)
        {
            switch (team)
            {
                case TrainingSource.MuJoCo:
                    return "MuJoCo";
                case TrainingSource.IsaacLab:
                    return "Isaac Lab";
                case TrainingSource.MlAgents:
                    return "ML-Agents";
                case TrainingSource.Heuristic:
                    return "Heuristic";
                default:
                    return "Unknown";
            }
        }

        /// <summary>
        /// Two-letter tag every racer name starts with, so a viewer can tell at a glance
        /// what trained its brain: MU (MuJoCo), IL (Isaac Lab), ML (ML-Agents), HC (hand-coded).
        /// </summary>
        public static string Tag(TrainingSource team)
        {
            switch (team)
            {
                case TrainingSource.MuJoCo:
                    return "MU";
                case TrainingSource.IsaacLab:
                    return "IL";
                case TrainingSource.MlAgents:
                    return "ML";
                case TrainingSource.Heuristic:
                    return "HC";
                default:
                    return string.Empty;
            }
        }

        /// <summary>A creature name with its trainer tag in front, e.g. "IL IsaacBox".</summary>
        public static string Tagged(TrainingSource team, string name)
        {
            string tag = Tag(team);
            return tag.Length == 0 ? name : tag + " " + name;
        }

        public static Color ColorOf(TrainingSource team)
        {
            switch (team)
            {
                case TrainingSource.MuJoCo:
                    return MuJoCo;
                case TrainingSource.IsaacLab:
                    return IsaacLab;
                case TrainingSource.MlAgents:
                    return MlAgents;
                case TrainingSource.Heuristic:
                    return Heuristic;
                default:
                    return Unknown;
            }
        }
    }
}
