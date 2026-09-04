using PoRacer.Agents;
using PoRacer.Views;
using UnityEngine;

namespace PoRacer.Systems
{
    /// <summary>
    /// Training-scene-only referee for one creature on one copy of an authored
    /// course: puts the creature on the centreline at the start of every episode,
    /// keeps its look-ahead goal riding the road, and ends the episode if it
    /// leaves the road - otherwise a policy learns to cut the switchbacks across
    /// the mountainside. Curriculum knobs from the trainer:
    ///   course_start_span   metres of course the start point is drawn from (0 = the start line)
    ///   course_lateral_span metres either side of the centreline the start is jittered by
    /// This component must never exist in a race scene.
    /// </summary>
    public sealed class Systems_CourseTrainingArea : MonoBehaviour
    {
        // 240 s of physics at the 0.005 s step: a course episode is long.
        private const int TRAINING_MAX_STEP = 48000;
        private const float DEFAULT_LATERAL_SPAN = 0.5f;
        private const float OFF_ROAD_MARGIN = 2.5f;
        private const float OFF_ROAD_REWARD = -1f;
        private const float GOAL_HEIGHT_CLEARANCE = 0.05f;

        [SerializeField] private Unity.MLAgents.Agent _agent;
        [SerializeField] private RaceCourseView _course;
        [SerializeField] private float _spawnHeight = 0.3f;

        private ICreatureAgent _creature;
        private CourseGoalView _goal;
        private ArticulationBody[] _bodies;
        private float _offRoadLimit;

        private void Awake()
        {
            _creature = (ICreatureAgent)_agent;
            _creature.MaxStep = TRAINING_MAX_STEP;
            var goalObject = new GameObject(name + ".goal");
            goalObject.transform.SetParent(transform, false);
            _goal = goalObject.AddComponent<CourseGoalView>();
            _goal.Initialize(_course.Path, _creature.Body);
            _creature.SetGoal(goalObject.transform);
            if (_agent is Agent_Creature creatureAgent)
            {
                creatureAgent.SetCourse(_goal);
            }
            _creature.SetAreaResetCallback(ResetArea);
            _bodies = _creature.Root.GetComponentsInChildren<ArticulationBody>();
            _offRoadLimit = _course.HalfWidth + OFF_ROAD_MARGIN;
        }

        private void FixedUpdate()
        {
            // Off the road is off the course: the projection would still count
            // progress across the hillside, and that is exactly the shortcut a
            // policy would find.
            if (_course.Path.LateralDistance(_creature.Body.position) > _offRoadLimit)
            {
                _agent.AddReward(OFF_ROAD_REWARD);
                _agent.EndEpisode();
            }
        }

        private void ResetArea()
        {
            float startSpan = 0f;
            float lateralSpan = DEFAULT_LATERAL_SPAN;
            if (Unity.MLAgents.Academy.IsInitialized)
            {
                var envParams = Unity.MLAgents.Academy.Instance.EnvironmentParameters;
                startSpan = Mathf.Clamp(envParams.GetWithDefault("course_start_span", 0f), 0f, _course.Path.Length - 5f);
                lateralSpan = Mathf.Clamp(envParams.GetWithDefault("course_lateral_span", DEFAULT_LATERAL_SPAN),
                    0f, _course.HalfWidth - 0.5f);
            }
            Systems_CoursePath path = _course.Path;
            float along = Random.Range(0f, startSpan);
            Vector3 heading = path.HeadingAt(along);
            Vector3 across = Vector3.Cross(Vector3.up, heading);
            Vector3 centre = path.PointAt(along);
            float lateral = Random.Range(-lateralSpan, lateralSpan);
            Vector3 surface = centre;
            for (int attempt = 0; attempt < 3; attempt++)
            {
                if (_course.TrySurfaceAt(centre + across * lateral, centre.y, out surface))
                {
                    break;
                }
                lateral *= 0.5f;
                surface = centre;
            }
            Vector3 position = surface + Vector3.up * (_spawnHeight + GOAL_HEIGHT_CLEARANCE);

            ArticulationBody root = _creature.Root;
            root.TeleportRoot(position, Quaternion.LookRotation(heading, Vector3.up) * _creature.RestRotation);
            ResetBody(root);
            for (int bodyIndex = 0; bodyIndex < _bodies.Length; bodyIndex++)
            {
                ResetBody(_bodies[bodyIndex]);
            }
            // The reward baselines on the distance the agent reads right after this
            // callback, so the carrot must already sit ahead of the new position.
            _goal.Refresh();
        }

        private static void ResetBody(ArticulationBody body)
        {
            if (body.jointPosition.dofCount > 0)
            {
                body.jointPosition = new ArticulationReducedSpace(0f);
                body.jointVelocity = new ArticulationReducedSpace(0f);
                ArticulationDrive drive = body.xDrive;
                drive.target = 0f;
                body.xDrive = drive;
            }
            body.linearVelocity = Vector3.zero;
            body.angularVelocity = Vector3.zero;
        }
    }
}
