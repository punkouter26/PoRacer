using System;
using Unity.MLAgents;
using UnityEngine;
using VContainer.Unity;

namespace PoRacer.Systems
{
    /// <summary>
    /// Applies display settings for the self-running app. Guarded so a build that
    /// is being driven by an ML-Agents trainer (communicator on) is never frame-capped
    /// — the trainer controls time scale and frame rate during training.
    /// </summary>
    public sealed class Systems_AppBootstrap : IStartable, IDisposable
    {
        private const int TARGET_FRAME_RATE = 60;

        public void Start()
        {
            if (Academy.IsInitialized && Academy.Instance.IsCommunicatorOn)
            {
                return;
            }
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = TARGET_FRAME_RATE;
        }

        public void Dispose() { }
    }
}
