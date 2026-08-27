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
            Screen.sleepTimeout = SleepTimeout.NeverSleep;
            Input.multiTouchEnabled = true;
            SuppressWarningStackTraces();
        }

        /// <summary>
        /// Drops the stack trace from Warning-level logs.
        ///
        /// URP re-warns once per shader constant array per rendered frame when its
        /// light arrays were sized for a different build target earlier in the same
        /// session ("exceeds previous array size ... Restart Unity"). At twelve
        /// arrays and 50+ fps that is ~700 entries a second, and with stack traces
        /// attached it wrote a 30 GB Editor.log and flushed every genuine error out
        /// of the 1000-entry console buffer. The warnings themselves stay; only the
        /// per-frame stack trace, which is always the same render-loop frames and
        /// never once told us anything, goes away.
        /// </summary>
        private static void SuppressWarningStackTraces()
        {
            Application.SetStackTraceLogType(LogType.Warning, StackTraceLogType.None);
        }

        public void Dispose() { }
    }
}
