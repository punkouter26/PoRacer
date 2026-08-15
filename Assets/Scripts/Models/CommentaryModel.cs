using System.Collections.Generic;

namespace PoRacer.Models
{
    /// <summary>
    /// Rolling race-commentary lines, newest first. Version bumps on every
    /// change so the HUD can refresh only when the text actually changed.
    /// </summary>
    public sealed class CommentaryModel
    {
        public const int MAX_LINES = 3;

        public readonly List<string> Lines = new();

        public int Version { get; private set; }

        public void Add(string line)
        {
            Lines.Insert(0, line);
            if (Lines.Count > MAX_LINES)
            {
                Lines.RemoveAt(Lines.Count - 1);
            }
            Version++;
        }

        public void Clear()
        {
            Lines.Clear();
            Version++;
        }
    }
}
