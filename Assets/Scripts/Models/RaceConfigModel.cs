using System;
using System.Collections.Generic;

namespace PoRacer.Models
{
    /// <summary>
    /// The pre-race selection: how many of each creature enter the race.
    /// MenuVisible drives whether the menu or the race HUD is on screen.
    /// </summary>
    public sealed class RaceConfigModel
    {
        public static readonly int[] COUNT_OPTIONS = { 0, 1, 10, 50, 100 };

        private readonly Dictionary<string, int> _counts = new();

        public bool MenuVisible = true;

        // Scripted mode races the coded gaits (no .onnx needed) instead of RL brains.
        public bool UseScriptedBrains;

        // Index into Systems_MapCatalog.Entries; the menu writes it, spawn reads it.
        public int SelectedMapIndex;

        public event Action Changed;

        public int GetCount(string creatureId)
        {
            return _counts.TryGetValue(creatureId, out int count) ? count : 0;
        }

        public void SetCount(string creatureId, int count)
        {
            _counts[creatureId] = count;
            Changed?.Invoke();
        }

        public int TotalCount()
        {
            int total = 0;
            foreach (KeyValuePair<string, int> pair in _counts)
            {
                total += pair.Value;
            }
            return total;
        }

        public void SetMap(int mapIndex)
        {
            SelectedMapIndex = mapIndex;
            Changed?.Invoke();
        }

        public void NotifyChanged() => Changed?.Invoke();
    }
}
