using System;
using System.Collections.Generic;
using MessagePipe;
using PoRacer.Models;
using VContainer;

namespace PoRacer.Systems
{
    /// <summary>
    /// Scores the Sim Wars league after every race: each racer's placing counts for the
    /// tool that trained its brain (RacerState.TrainedBy). Podium places earn 5/3/1 points;
    /// a racer that crossed first is a win for its team. The MuJoCo vs Isaac Lab
    /// head-to-head goes to whichever tool's best finisher placed higher, in races both
    /// entered.
    ///
    /// This is entertainment, not the comparison: the two tools race different bodies
    /// here, so a win says as much about the creature as about the trainer. The fair
    /// comparison is one body under one contract, judged by the shared Walking Standard
    /// (AGENTS.md rule J, docs/Plan-TrainingMethodComparison.md), and the league view says so.
    /// </summary>
    public sealed class Systems_SimWars : IDisposable
    {
        public const string FILE_NAME = "sim-wars.json";
        private const int NOT_PLACED = int.MaxValue;

        private static readonly int[] PodiumPoints = { 5, 3, 1 };

        [Serializable]
        private sealed class TeamRow
        {
            public int team;
            public int races;
            public int wins;
            public int podiums;
            public int points;
        }

        [Serializable]
        private sealed class LeagueFile
        {
            public int racesScored;
            public int mujocoAhead;
            public int isaacLabAhead;
            public int draws;
            public List<TeamRow> teams = new();
        }

        private readonly SimWarsModel _model;
        private readonly RaceModel _raceModel;
        private readonly Systems_Persistence _persistence;
        private readonly IDisposable _subscription;
        private readonly int[] _bestPlaceByTeam;
        private readonly bool[] _enteredByTeam;
        private readonly Func<string, TrainingSource> _teamOfRacer;

        [Inject]
        public Systems_SimWars(SimWarsModel model, RaceModel raceModel, Systems_Persistence persistence,
            ISubscriber<RaceFinishedMessage> raceFinished)
            : this(model)
        {
            _raceModel = raceModel;
            _persistence = persistence;
            _teamOfRacer = TeamOfRacer;
            Load();
            _subscription = raceFinished.Subscribe(OnRaceFinished);
        }

        // Test constructor: league maths only, no messaging or persistence.
        public Systems_SimWars(SimWarsModel model)
        {
            _model = model;
            int teamCount = Enum.GetValues(typeof(TrainingSource)).Length;
            _bestPlaceByTeam = new int[teamCount];
            _enteredByTeam = new bool[teamCount];
        }

        public void Dispose() => _subscription?.Dispose();

        /// <summary>Scores one race. <paramref name="teamOf"/> maps a racer id to its team.</summary>
        public void ApplyResults(IReadOnlyList<RaceResultEntry> results, Func<string, TrainingSource> teamOf)
        {
            for (int teamIndex = 0; teamIndex < _bestPlaceByTeam.Length; teamIndex++)
            {
                _bestPlaceByTeam[teamIndex] = NOT_PLACED;
                _enteredByTeam[teamIndex] = false;
                _model.Team((TrainingSource)teamIndex).LastRacePoints = 0;
            }

            for (int resultIndex = 0; resultIndex < results.Count; resultIndex++)
            {
                RaceResultEntry result = results[resultIndex];
                TrainingSource team = teamOf(result.RacerId);
                int teamIndex = (int)team;
                _enteredByTeam[teamIndex] = true;
                // A DNF can be handed a podium slot when too few crossed; it still did
                // not finish, so it scores nothing and never counts as a team's best.
                if (result.Dnf || result.Place <= 0)
                {
                    continue;
                }
                if (result.Place < _bestPlaceByTeam[teamIndex])
                {
                    _bestPlaceByTeam[teamIndex] = result.Place;
                }
                if (result.Place <= PodiumPoints.Length)
                {
                    SimWarsTeamStats stats = _model.Team(team);
                    int points = PodiumPoints[result.Place - 1];
                    stats.Points += points;
                    stats.LastRacePoints += points;
                    stats.Podiums++;
                    if (result.Place == 1)
                    {
                        stats.Wins++;
                    }
                }
            }

            for (int teamIndex = 0; teamIndex < _enteredByTeam.Length; teamIndex++)
            {
                if (_enteredByTeam[teamIndex])
                {
                    _model.Team((TrainingSource)teamIndex).Races++;
                }
            }

            int mujoco = (int)TrainingSource.MuJoCo;
            int isaac = (int)TrainingSource.IsaacLab;
            if (_enteredByTeam[mujoco] && _enteredByTeam[isaac])
            {
                if (_bestPlaceByTeam[mujoco] < _bestPlaceByTeam[isaac])
                {
                    _model.MuJoCoAhead++;
                }
                else if (_bestPlaceByTeam[isaac] < _bestPlaceByTeam[mujoco])
                {
                    _model.IsaacLabAhead++;
                }
                else
                {
                    // Both unplaced: neither tool got a racer home.
                    _model.HeadToHeadDraws++;
                }
            }
            _model.RacesScored++;
            _model.Version++;
        }

        private void OnRaceFinished(RaceFinishedMessage message)
        {
            ApplyResults(message.Results, _teamOfRacer);
            Save();
        }

        private TrainingSource TeamOfRacer(string racerId)
        {
            RacerState racer = _raceModel.FindRacer(racerId);
            return racer != null ? racer.TrainedBy : TrainingSource.Unknown;
        }

        private void Load()
        {
            LeagueFile file = _persistence.Load<LeagueFile>(FILE_NAME);
            _model.Reset();
            _model.RacesScored = file.racesScored;
            _model.MuJoCoAhead = file.mujocoAhead;
            _model.IsaacLabAhead = file.isaacLabAhead;
            _model.HeadToHeadDraws = file.draws;
            int teamCount = _bestPlaceByTeam.Length;
            for (int rowIndex = 0; rowIndex < file.teams.Count; rowIndex++)
            {
                TeamRow row = file.teams[rowIndex];
                if (row.team < 0 || row.team >= teamCount)
                {
                    continue;
                }
                SimWarsTeamStats stats = _model.Team((TrainingSource)row.team);
                stats.Races = row.races;
                stats.Wins = row.wins;
                stats.Podiums = row.podiums;
                stats.Points = row.points;
            }
            _model.Version++;
        }

        private void Save()
        {
            var file = new LeagueFile
            {
                racesScored = _model.RacesScored,
                mujocoAhead = _model.MuJoCoAhead,
                isaacLabAhead = _model.IsaacLabAhead,
                draws = _model.HeadToHeadDraws
            };
            for (int teamIndex = 0; teamIndex < _bestPlaceByTeam.Length; teamIndex++)
            {
                SimWarsTeamStats stats = _model.Team((TrainingSource)teamIndex);
                if (stats.Races == 0)
                {
                    continue;
                }
                file.teams.Add(new TeamRow
                {
                    team = teamIndex,
                    races = stats.Races,
                    wins = stats.Wins,
                    podiums = stats.Podiums,
                    points = stats.Points
                });
            }
            _persistence.Save(FILE_NAME, file);
        }
    }
}
