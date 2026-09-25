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
        /// <summary>
        /// Per-creature counts offered on the roster row. FOUR, not the five this
        /// carried until 2026-09-10, and the count is a layout constraint rather
        /// than a taste: each one is a finger target on a single row of a 420 dp
        /// panel, and Android's minimum is 48 dp. Five cells measured 42 x 60 dp
        /// on the racer screen — 36 dp on a 360 dp handset — because the segment
        /// track takes a fixed 56% of the row and five shares of that is all there
        /// is. Four shares is 48 dp and clears the minimum with the name and ELO
        /// columns intact.
        ///
        /// Small counts since 2026-09-25: a field of a few of each creature is the
        /// race people watch, and the presets row above the roster ("All x1",
        /// "All x2") covers bulk selection. Adding a fifth option back means finding
        /// it 48 dp somewhere else on the row first.
        /// </summary>
        public static readonly int[] COUNT_OPTIONS = { 0, 1, 2, 5 };

        private readonly Dictionary<string, int> _counts = new();

        public bool MenuVisible = true;

        // Index into Systems_MapCatalog.Entries; the menu writes it, spawn reads it.
        public int SelectedMapIndex;

        // Random per-racer power/mass quirks (TURBO, HEAVY...). Off for the walking
        // exam: a +12 % drive or +12 % mass is a handicap the brain did not earn.
        public bool QuirksEnabled = true;

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
