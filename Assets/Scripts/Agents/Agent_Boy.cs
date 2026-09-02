using System;
using Boy;
using UnityEngine;

namespace PoRacer.Agents
{
    /// <summary>
    /// Race adapter for the Isaac Lab Boy: exposes <see cref="BoyAgent"/> (Inference Engine,
    /// no ML-Agents) through <see cref="ICreatureAgent"/> so the spawner, RacerView and camera
    /// treat it like any catalog racer. Lives on the same GameObject as the Isaac agent.
    ///
    /// Runs after <see cref="BoyAgent"/> (execution order 100) so that its Start - which builds
    /// the inference worker - has already run before this component parks it.
    /// </summary>
    [RequireComponent(typeof(BoyAgent))]
    [DisallowMultipleComponent]
    [DefaultExecutionOrder(100)]
    public sealed class Agent_Boy : MonoBehaviour, ICreatureAgent
    {
        [Tooltip("Pin the root until something solid is under it. SCN_RACE_FLAT has no ground " +
                 "in edit mode - Systems_TrackBuilder raises the track when the race starts.")]
        [SerializeField] private bool _holdUntilGrounded = true;

        [SerializeField] private float _groundProbeDistance = 4f;

        [Tooltip("dot(root.up, world.up) below this counts as down. 0.5 is 60 degrees off vertical.")]
        [SerializeField] private float _fallenUprightDot = 0.5f;

        [Tooltip("How long it must stay down before RacerView is told it Failed.")]
        [SerializeField] private float _fallenGraceSeconds = 1f;

        private BoyAgent _agent;
        private BoyTargetSampler _sampler;
        private Action _areaReset;
        private readonly RaycastHit[] _probeHits = new RaycastHit[8];
        private bool _held;
        private float _fallenFor;

        public bool Failed => _fallenFor >= _fallenGraceSeconds;

        public ArticulationBody Root => _agent != null ? _agent.Root : GetComponentInChildren<ArticulationBody>();

        /// <summary>The hips link: the prefab root is an inert container.</summary>
        public Transform Body => Root != null ? Root.transform : transform;

        public int MaxStep { get; set; }

        /// <summary>Authored upright (T-pose zero, world-aligned links), so the rest pose is the identity.</summary>
        public Quaternion RestRotation => Quaternion.identity;

        private void Awake()
        {
            _agent = GetComponent<BoyAgent>();
            _sampler = GetComponent<BoyTargetSampler>();
            // RacerView owns rescue and retire for racers; the agent's own fall recovery would
            // teleport a racer the marshal is already handling.
            _agent.autoRecoverFromFalls = false;
            _agent.showOnGuiReadout = false;
        }

        private void Start()
        {
            if (!_holdUntilGrounded)
            {
                return;
            }
            ArticulationBody root = Root;
            if (root == null)
            {
                return;
            }
            _held = true;
            root.immovable = true;
            _agent.enabled = false;
        }

        private void FixedUpdate()
        {
            if (_held)
            {
                if (ProbeGround())
                {
                    Release();
                }
                return;
            }
            ArticulationBody root = Root;
            if (root == null)
            {
                return;
            }
            bool down = Vector3.Dot(root.transform.up, Vector3.up) < _fallenUprightDot;
            _fallenFor = down ? _fallenFor + Time.fixedDeltaTime : 0f;
        }

        public void SetGoal(Transform goal)
        {
            _agent.target = goal;
            // An explicit target takes priority over the sampler; switch the sampler off so a
            // grid of racers does not keep sampling rings into the void.
            if (goal != null && _sampler != null)
            {
                _sampler.enabled = false;
            }
        }

        public void SetAreaResetCallback(Action areaReset) => _areaReset = areaReset;

        /// <summary>
        /// Deliberately does nothing. The Boy has no fatigue system. Calling
        /// <see cref="BoyAgent.Reconfigure"/> here would rewrite every xDrive from the rig asset
        /// and wipe the stiffness and forceLimit a spawn quirk just wrote; the agent's per-tick
        /// drive write copies the existing ArticulationDrive and changes only its target.
        /// </summary>
        public void NotifyDrivesChanged()
        {
        }

        private void Release()
        {
            _held = false;
            _fallenFor = 0f;
            ArticulationBody root = Root;
            if (root != null)
            {
                root.immovable = false;
            }
            _agent.enabled = true;
        }

        private bool ProbeGround()
        {
            ArticulationBody root = Root;
            if (root == null)
            {
                return false;
            }
            int count = Physics.RaycastNonAlloc(root.transform.position, Vector3.down, _probeHits,
                                                _groundProbeDistance, ~0, QueryTriggerInteraction.Ignore);
            for (int hitIndex = 0; hitIndex < count; hitIndex++)
            {
                if (!_probeHits[hitIndex].collider.transform.IsChildOf(transform))
                {
                    return true;
                }
            }
            return false;
        }
    }
}
