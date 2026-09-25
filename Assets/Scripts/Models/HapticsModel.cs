using System;

namespace PoRacer.Models
{
    /// <summary>
    /// Whether the phone vibrates on race moments. The menu toggles it,
    /// Systems_Haptics remembers it, HapticsView obeys it.
    /// </summary>
    public sealed class HapticsModel
    {
        public bool Enabled { get; private set; } = true;

        public event Action Changed;

        public void SetEnabled(bool enabled)
        {
            if (enabled == Enabled)
            {
                return;
            }
            Enabled = enabled;
            Changed?.Invoke();
        }
    }
}
