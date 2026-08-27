using UnityEditor;
using UnityEngine;

namespace PoRacer.Editor
{
    /// <summary>
    /// Drops the stack trace from Warning-level logs in the Editor.
    ///
    /// URP sizes its light and reflection-probe shader constant arrays once per
    /// editor session, from the active build target. Switching target inside a
    /// live session (this project builds both an Android player and Windows
    /// training envs) leaves them at the old size, and URP then re-warns
    /// "exceeds previous array size ... Restart Unity to recreate the arrays"
    /// once per array per rendered frame - twelve arrays, every Scene View
    /// repaint, forever.
    ///
    /// On 2026-08-27 that had grown Logs/Editor.log to 31.9 GB, and because each
    /// warning carried a ~25-frame render-loop stack trace it was also flushing
    /// the 1000-entry console buffer roughly twice a second, which hid every
    /// genuine error behind noise.
    ///
    /// Dropping the trace is the part worth keeping: the trace is identical every
    /// time and names only URP and UIElements internals. The warning itself still
    /// prints, so the "restart Unity" advice is not lost - and restarting really
    /// is the fix, since the arrays cannot be resized once allocated.
    ///
    /// Systems_AppBootstrap does the same for play mode and players.
    /// </summary>
    [InitializeOnLoad]
    internal static class Editor_LogHygiene
    {
        static Editor_LogHygiene()
        {
            Application.SetStackTraceLogType(LogType.Warning, StackTraceLogType.None);
        }
    }
}
