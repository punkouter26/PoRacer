using System;
using Creature;
using Mujoco;
using UnityEngine;

namespace PoRacer.Agents
{
    /// <summary>
    /// Race adapter for Fido, the MuJoCo quadruped: exposes <see cref="CreatureAgent"/>
    /// through <see cref="ICreatureAgent"/> so the spawner, RacerView and camera treat
    /// him like any other catalog racer. Lives on the imported MJCF root.
    ///
    /// Fido is the one racer PhysX does not simulate. His bodies are MuJoCo's, stepped
    /// by mujoco.dll inside the scene's single <see cref="MjScene"/>, which writes the
    /// results back onto Unity transforms every FixedUpdate. Three consequences the rest
    /// of the code has to live with:
    ///
    ///   * He has no ArticulationBody, so <see cref="Root"/> is null. Systems_Spawn and
    ///     RacerView both read a null Root as "track the prefab root instead", which is
    ///     the right answer here.
    ///   * He collides with the ground plane and with other Fidos (same MuJoCo world),
    ///     but not with PhysX racers or the track's trigger volumes. The mud, boost and
    ///     gust zones key off `other.attachedArticulationBody` and so skip him rather
    ///     than throw.
    ///   * His policy takes no goal input — 33 observations of his own body and nothing
    ///     about the world — so he runs straight ahead and cannot steer to the finish
    ///     line. <see cref="SetGoal"/> is deliberately a no-op.
    /// </summary>
    [RequireComponent(typeof(CreatureAgent))]
    [DisallowMultipleComponent]
    public sealed class Agent_Fido : MonoBehaviour, ICreatureAgent, IMujocoCreature
    {
        /// <summary>
        /// The MJCF builds Fido along +X (his head sits at x = +0.21) and the importer
        /// maps MuJoCo (x, y, z) onto Unity (x, z, y), so he imports facing Unity +X.
        /// PoRacer races down +Z. This is the yaw that closes the gap; the prefab root
        /// is authored with it, and RacerView re-applies it under the heading whenever
        /// it stands a flipped racer back up.
        /// </summary>
        private static readonly Quaternion Rest = Quaternion.Euler(0f, -90f, 0f);

        // Fido's torso rides at 0.34 m standing and dipped to 0.245 m at the bottom of
        // his gait in the drop's own measurements, so 0.15 m is well clear of walking.
        private const float FALLEN_HEIGHT = 0.15f;
        private const float FALLEN_UPRIGHT_DOT = 0.25f;
        private const float FALLEN_GRACE_SECONDS = 1.5f;

        /// <summary>
        /// Torso height the policy was trained to start from, above the MuJoCo ground
        /// plane that Systems_MujocoWorld puts at y = 0. It is the MJCF's own torso pos.
        /// </summary>
        private const float TRAINED_TORSO_HEIGHT = 0.34f;

        private CreatureAgent _creature;
        private Transform _torso;
        private float _fallenFor;
        private bool _failed;

        public bool Failed => _failed;

        /// <summary>Always null: MuJoCo simulates Fido, so no ArticulationBody exists.</summary>
        public ArticulationBody Root => null;

        /// <summary>
        /// The torso, which carries the MJCF free joint and is therefore the one thing
        /// MuJoCo actually moves. The imported container this component sits on never
        /// leaves the start line, so it is emphatically not the transform to track.
        /// </summary>
        public Transform Body => _torso != null ? _torso : transform;

        public int MaxStep { get; set; }

        public Quaternion RestRotation => Rest;

        private void Awake()
        {
            _creature = GetComponent<CreatureAgent>();
            BindToOwnHierarchy();
            _torso = _creature.torso != null ? _creature.torso.transform : transform;
            // Eight racers logging their bindings and home pose would bury the console.
            _creature.logBindings = false;
            SnapToTrainedStance();
        }

        /// <summary>
        /// Drops the racer onto the ground at exactly the ride height the policy trained
        /// from, cancelling the spawner's deliberate few-centimetre drop.
        ///
        /// That drop is there so a PhysX creature is never born intersecting the ground,
        /// and for a ragdoll it costs nothing — it lands and carries on. Fido is a trained
        /// policy, and 5 cm is enough to put him outside the state distribution he ever
        /// saw: measured, a torso starting at 0.39 m instead of 0.34 m collapses on the
        /// first stride, every time, while 0.34 m walks away cleanly. He is also immune to
        /// the problem the drop solves, because MuJoCo resolves his contacts rather than
        /// PhysX.
        ///
        /// This runs in Awake, so it lands before MjScene compiles the model in Start —
        /// CreateScene reads Unity transforms, so the corrected pose is what MuJoCo gets.
        /// </summary>
        private void SnapToTrainedStance()
        {
            if (_torso == null)
            {
                return;
            }
            // The torso's authored height inside the prefab, whatever the root is at.
            float torsoOffset = _torso.position.y - transform.position.y;
            Vector3 position = transform.position;
            position.y = TRAINED_TORSO_HEIGHT - torsoOffset;
            transform.position = position;
        }

        private void FixedUpdate()
        {
            if (_failed || _torso == null)
            {
                return;
            }
            bool down = _torso.position.y < FALLEN_HEIGHT
                || Vector3.Dot(_torso.up, Vector3.up) < FALLEN_UPRIGHT_DOT;
            _fallenFor = down ? _fallenFor + Time.fixedDeltaTime : 0f;
            if (_fallenFor >= FALLEN_GRACE_SECONDS)
            {
                _failed = true;
            }
        }

        /// <summary>
        /// Fido's policy observes only his own body, so there is nothing to aim.
        /// </summary>
        public void SetGoal(Transform goal)
        {
        }

        public void SetAreaResetCallback(Action areaReset)
        {
        }

        /// <summary>
        /// No-op: joint power is MuJoCo actuator gear, not an ArticulationDrive, so the
        /// spawner's power quirk never touches Fido and there is no baseline to recapture.
        /// </summary>
        public void NotifyDrivesChanged()
        {
        }

        /// <summary>
        /// Fills CreatureAgent's torso/joint/actuator slots from this instance's own
        /// children.
        ///
        /// CreatureAgent auto-resolves those slots with FindObjectsByType, which searches
        /// the entire scene. That is correct for the single-creature verification scene it
        /// shipped with, but a race fields several Fidos inside one MjScene: each would
        /// resolve to whichever Fido the scene happened to return first, and every brain
        /// would drive that one dog while the rest stood inert. Binding here, in Awake and
        /// so before the first control callback, keeps each Fido on his own actuators.
        /// </summary>
        private void BindToOwnHierarchy()
        {
            if (_creature.torso == null)
            {
                _creature.torso = GetComponentInChildren<MjBody>(true);
            }

            if (_creature.actuators.Count == 0)
            {
                MjActuator[] mine = GetComponentsInChildren<MjActuator>(true);
                for (int orderIndex = 0; orderIndex < CreatureAgent.ActuatorOrder.Length; orderIndex++)
                {
                    MjActuator match = MatchByName(mine, CreatureAgent.ActuatorOrder[orderIndex]);
                    if (match == null)
                    {
                        Debug.LogError(
                            $"Agent_Fido: no MjActuator named '{CreatureAgent.ActuatorOrder[orderIndex]}' " +
                            "under this racer; leaving the slots empty so CreatureAgent reports it.", this);
                        _creature.actuators.Clear();
                        break;
                    }
                    _creature.actuators.Add(match);
                }
            }

            if (_creature.joints.Count == 0)
            {
                // The MJCF gives each hinge the same name as the motor that drives it.
                MjHingeJoint[] mine = GetComponentsInChildren<MjHingeJoint>(true);
                for (int orderIndex = 0; orderIndex < CreatureAgent.ActuatorOrder.Length; orderIndex++)
                {
                    MjHingeJoint match = MatchByName(mine, CreatureAgent.ActuatorOrder[orderIndex]);
                    if (match == null)
                    {
                        Debug.LogError(
                            $"Agent_Fido: no MjHingeJoint named '{CreatureAgent.ActuatorOrder[orderIndex]}' " +
                            "under this racer; leaving the slots empty so CreatureAgent reports it.", this);
                        _creature.joints.Clear();
                        break;
                    }
                    _creature.joints.Add(match);
                }
            }
        }

        /// <summary>
        /// Matches on the MuJoCo name, the GameObject name, or either with a trailing
        /// "_&lt;digits&gt;" stripped — MjScene.CreateScene uniquifies names by appending
        /// an id, turning "fl_hip" into "fl_hip_30". Binding runs before CreateScene, so
        /// the plain names normally hit; the stripping is there so a re-bind after a
        /// scene rebuild still resolves.
        /// </summary>
        private static T MatchByName<T>(T[] candidates, string wanted) where T : MjComponent
        {
            for (int candidateIndex = 0; candidateIndex < candidates.Length; candidateIndex++)
            {
                T candidate = candidates[candidateIndex];
                if (candidate.MujocoName == wanted || candidate.gameObject.name == wanted)
                {
                    return candidate;
                }
            }
            for (int candidateIndex = 0; candidateIndex < candidates.Length; candidateIndex++)
            {
                T candidate = candidates[candidateIndex];
                if (BaseName(candidate.MujocoName) == wanted || BaseName(candidate.gameObject.name) == wanted)
                {
                    return candidate;
                }
            }
            return null;
        }

        private static string BaseName(string name)
        {
            if (string.IsNullOrEmpty(name))
            {
                return name;
            }
            int split = name.LastIndexOf('_');
            if (split <= 0 || split == name.Length - 1)
            {
                return name;
            }
            for (int charIndex = split + 1; charIndex < name.Length; charIndex++)
            {
                if (!char.IsDigit(name[charIndex]))
                {
                    return name;
                }
            }
            return name.Substring(0, split);
        }
    }
}
