using System;
using System.Collections.Generic;
using IsaacBox;
using UnityEngine;

namespace PoRacer.Agents
{
    /// <summary>
    /// Race adapter for the Isaac Lab IsaacBox: exposes <see cref="IsaacBoxAgent"/> (Inference Engine,
    /// no ML-Agents) through <see cref="ICreatureAgent"/> so the spawner, RacerView and camera
    /// treat it like any catalog racer. Lives on the same GameObject as the Isaac agent.
    ///
    /// Runs after <see cref="IsaacBoxAgent"/> (execution order 100) so that its Start - which builds
    /// the inference worker - has already run before this component parks it.
    /// </summary>
    [RequireComponent(typeof(IsaacBoxAgent))]
    [DisallowMultipleComponent]
    [DefaultExecutionOrder(100)]
    public sealed class Agent_IsaacBox : MonoBehaviour, ICreatureAgent, IPolicyReadout, IAuthoredAppearance
    {
        [Tooltip("Pin the root until something solid is under it. SCN_RACE_FLAT has no ground " +
                 "in edit mode - Systems_TrackBuilder raises the track when the race starts.")]
        [SerializeField] private bool _holdUntilGrounded = true;

        [SerializeField] private float _groundProbeDistance = 4f;

        private IsaacBoxAgent _agent;
        private IsaacBoxTargetSampler _sampler;
        private readonly RaycastHit[] _probeHits = new RaycastHit[8];
        private bool _held;
        private bool _startHeld;
        private bool _failed;

        /// <summary>
        /// A physics failure he cannot come back from, and nothing else. Falls go to
        /// RacerView's 12 s knockdown referee like every other racer (AGENTS rule H);
        /// see Agent_IsaacH1.Failed for why the old 1 s fall check had to go.
        /// </summary>
        public bool Failed => _failed;

        /// <summary>
        /// Held by parking the policy and pinning the base, which is the same pair
        /// <see cref="Start"/> already uses for the grounding hold and for the same
        /// reason: he is a humanoid, and a humanoid whose balance controller is
        /// switched off for 2.4 s is on the floor before GO. Pinned, he cannot fall.
        ///
        /// The two holds are independent and both have to clear before he runs, so
        /// the grounding probe and the countdown cannot release each other early.
        /// </summary>
        public bool StartHeld
        {
            get => _startHeld;
            set
            {
                if (_startHeld == value)
                {
                    return;
                }
                _startHeld = value;
                ApplyStartGate();
            }
        }

        public ArticulationBody Root => _agent != null ? _agent.Root : GetComponentInChildren<ArticulationBody>();

        /// <summary>The hips link: the prefab root is an inert container.</summary>
        public Transform Body => Root != null ? Root.transform : transform;

        public int MaxStep { get; set; }

        /// <summary>The Isaac policy's raw output from its last decision; unclamped, so it can overshoot 1.</summary>
        public IReadOnlyList<float> LastActions => _agent != null ? _agent.LatestAction : null;

        /// <summary>Authored upright (T-pose zero, world-aligned links), so the rest pose is the identity.</summary>
        public Quaternion RestRotation => Quaternion.identity;

        private void Awake()
        {
            _agent = GetComponent<IsaacBoxAgent>();
            _sampler = GetComponent<IsaacBoxTargetSampler>();
            // RacerView owns rescue and retire for racers; the agent's own fall recovery would
            // teleport a racer the marshal is already handling.
            _agent.autoRecoverFromFalls = false;
            _agent.showOnGuiReadout = false;
            // Not an ML-Agents agent, so the debug overlay cannot read this off a
            // BehaviorParameters like it does for the other six.
            Agent_NativeBrainProbe.Attach(gameObject, "IsaacBox", "IE/Burst",
                _agent.rig != null ? _agent.rig.obsDim : 0,
                _agent.rig != null ? _agent.rig.actDim : 0,
                _agent.rig != null);
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
            if (_startHeld)
            {
                return;
            }
            ArticulationBody root = Root;
            if (root == null)
            {
                return;
            }
            Vector3 position = root.transform.position;
            if (!float.IsFinite(position.x) || !float.IsFinite(position.y) || !float.IsFinite(position.z))
            {
                _failed = true;
            }
        }

        /// <summary>
        /// Puts the start gate into effect, unless the grounding hold owns him — that
        /// one released through <see cref="Release"/>, which consults the gate itself.
        /// </summary>
        private void ApplyStartGate()
        {
            if (_held)
            {
                return;
            }
            _agent.enabled = !_startHeld;
            ArticulationBody root = Root;
            if (root != null)
            {
                root.immovable = _startHeld;
            }
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

        /// <summary>
        /// Deliberately does nothing. The IsaacBox has no fatigue system. Calling
        /// <see cref="IsaacBoxAgent.Reconfigure"/> here would rewrite every xDrive from the rig asset
        /// and wipe the stiffness and forceLimit a spawn quirk just wrote; the agent's per-tick
        /// drive write copies the existing ArticulationDrive and changes only its target.
        /// </summary>
        public void NotifyDrivesChanged()
        {
        }

        private void Release()
        {
            _held = false;
            // Hands him to the start gate rather than straight to the policy: ground
            // under his feet is permission to stand, not permission to race.
            ApplyStartGate();
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
