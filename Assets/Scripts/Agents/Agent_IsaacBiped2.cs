using System;
using IsaacBiped2;
using UnityEngine;

namespace PoRacer.Agents
{
    /// <summary>
    /// Race adapter for the Isaac Lab biped: exposes <see cref="IsaacBiped2Agent"/> (Inference
    /// Engine, no ML-Agents) through <see cref="ICreatureAgent"/> so the spawner, RacerView and
    /// camera treat it like any catalog racer. Lives on the same GameObject as the Isaac agent.
    ///
    /// Runs after <see cref="IsaacBiped2Agent"/> (execution order 100) so its Awake — which builds
    /// the inference worker — has already run before this component parks it, exactly as the H1
    /// adapter does. Without the ordering every racer would build its worker at the moment the
    /// field is released and the start would hitch.
    /// </summary>
    [RequireComponent(typeof(IsaacBiped2Agent))]
    [DisallowMultipleComponent]
    [DefaultExecutionOrder(100)]
    public sealed class Agent_IsaacBiped2 : MonoBehaviour, ICreatureAgent
    {
        [Tooltip("dot(root.up, world.up) below this counts as down. 0.5 is 60 degrees off vertical.")]
        [SerializeField] private float _fallenUprightDot = 0.5f;

        [Tooltip("How long it must stay down before RacerView is told it Failed. Stops a stumble " +
                 "through the vertical from retiring a racer that recovers on the next stride.")]
        [SerializeField] private float _fallenGraceSeconds = 1f;

        private IsaacBiped2Agent _agent;
        private Action _areaReset;
        private float _fallenFor;

        public bool Failed => _fallenFor >= _fallenGraceSeconds;

        public ArticulationBody Root => _agent != null ? _agent.Root : GetComponentInChildren<ArticulationBody>();

        /// <summary>Articulation root's transform; the prefab root only when there is no articulation.</summary>
        public Transform Body => Root != null ? Root.transform : transform;

        public int MaxStep { get; set; }

        /// <summary>The biped is authored upright, so its rest pose is the identity.</summary>
        public Quaternion RestRotation => Quaternion.identity;

        private void Awake()
        {
            _agent = GetComponent<IsaacBiped2Agent>();
            // RacerView owns rescue and retire for racers; the agent's own fall recovery would
            // teleport a racer the marshal is already handling.
            _agent.autoRecoverFromFalls = false;
            _agent.showOnGuiReadout = false;
        }

        private void FixedUpdate()
        {
            ArticulationBody root = Root;
            if (root == null)
            {
                return;
            }
            bool down = Vector3.Dot(root.transform.up, Vector3.up) < _fallenUprightDot;
            _fallenFor = down ? _fallenFor + Time.fixedDeltaTime : 0f;
        }

        public void SetGoal(Transform goal) => _agent.target = goal;

        public void SetAreaResetCallback(Action areaReset) => _areaReset = areaReset;

        /// <summary>
        /// Deliberately does nothing, for the same reason as the H1 adapter: the ML-Agents
        /// creatures re-capture a fatigue baseline here, but this agent has no fatigue system.
        /// Re-applying drives would overwrite the stiffness and forceLimit that
        /// Systems_Spawn.ApplyQuirk just wrote. The quirk survives on its own, because the agent's
        /// per-tick drive write copies the existing ArticulationDrive and changes only its target.
        /// </summary>
        public void NotifyDrivesChanged()
        {
        }
    }
}
