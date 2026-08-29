using System;
using IsaacSpider;
using UnityEngine;

namespace PoRacer.Agents
{
    /// <summary>
    /// Race adapter for the Isaac Lab spider: exposes <see cref="IsaacSpiderAgent"/> (Inference Engine,
    /// no ML-Agents) through <see cref="ICreatureAgent"/> so the spawner, RacerView and camera treat it
    /// like any catalog racer. Lives on the same GameObject as the Isaac agent.
    /// </summary>
    [RequireComponent(typeof(IsaacSpiderAgent))]
    [DisallowMultipleComponent]
    public sealed class Agent_IsaacSpider : MonoBehaviour, ICreatureAgent
    {
        private IsaacSpiderAgent _agent;
        private Action _areaReset;

        public bool Failed => _agent != null && _agent.IsFlipped;
        public ArticulationBody Root => _agent != null ? _agent.Root : GetComponentInChildren<ArticulationBody>();

        /// <summary>Articulation root's transform; the prefab root only when there is no articulation.</summary>
        public Transform Body => Root != null ? Root.transform : transform;
        public int MaxStep { get; set; }
        public Quaternion RestRotation => Quaternion.identity;

        private void Awake()
        {
            _agent = GetComponent<IsaacSpiderAgent>();
            // RacerView owns rescue/retire for racers; the agent's own flip reset would fight it.
            _agent.AutoRecover = false;
            _agent.ShowGui = false;
        }

        public void SetGoal(Transform goal) => _agent.SetTarget(goal);

        public void SetAreaResetCallback(Action areaReset) => _areaReset = areaReset;

        public void NotifyDrivesChanged() => _agent.NotifyDrivesChanged();
    }
}
