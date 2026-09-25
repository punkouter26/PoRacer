using System;
using System.IO;
using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// Writes a report object as indented JSON under Logs/. In the editor that is the
    /// project's own Logs/ folder (next to Assets/), where the agents already look for
    /// smoke_*.json and walkexam_*.json; in a player it falls back to persistentDataPath.
    /// Written to a temp file first and moved over, so a reader never sees half a report.
    /// </summary>
    public static class CreatureReportWriter
    {
        public static string Write(object report, string fileStem)
        {
            string directory = LogDirectory();
            Directory.CreateDirectory(directory);
            string path = Path.Combine(directory, fileStem + ".json");
            string temp = path + ".tmp";
            File.WriteAllText(temp, JsonUtility.ToJson(report, true));
            if (File.Exists(path))
            {
                File.Delete(path);
            }
            File.Move(temp, path);
            return path;
        }

        public static string Stamp()
        {
            return DateTime.Now.ToString("yyyyMMdd_HHmmss");
        }

        public static string Now()
        {
            return DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        }

        private static string LogDirectory()
        {
#if UNITY_EDITOR
            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            return Path.Combine(projectRoot, "Logs");
#else
            return Path.Combine(Application.persistentDataPath, "Logs");
#endif
        }
    }
}
