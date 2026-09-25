using System;

namespace PoRacer.Models
{
    /// <summary>
    /// The sky the scene is lit by right now. Systems_Sky writes it; PostFxView
    /// applies it and the race intro card names it. Null until Systems_Sky first
    /// picks one.
    /// </summary>
    public sealed class SkyModel
    {
        public SkyPreset Current { get; private set; }

        public event Action Changed;

        public void Set(SkyPreset preset)
        {
            if (preset == Current)
            {
                return;
            }
            Current = preset;
            Changed?.Invoke();
        }
    }
}
