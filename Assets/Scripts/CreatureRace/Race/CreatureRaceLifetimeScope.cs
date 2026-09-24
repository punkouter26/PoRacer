using UnityEngine;
using VContainer;
using VContainer.Unity;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// Composition root of a MuJoCo-only creature race scene (SCN_QUAD_RACE, and the next
    /// creature trained on MuJoCo Warp): the shared race bindings plus the generic MuJoCo
    /// spawner. Every class asks for exactly what it uses (architecture.md: no GameContext).
    ///
    /// Note for whoever copies this file: VContainer's ScriptTemplateProcessor replaces any
    /// NEW *LifetimeScope.cs with its empty template the moment Unity creates the .meta.
    /// Write the file, let Unity import it, then write the real content again.
    /// </summary>
    public sealed class CreatureRaceLifetimeScope : LifetimeScope
    {
        [Tooltip("The creature's CreatureRaceSettings asset, wired by its scene builder.")]
        [SerializeField] private CreatureRaceSettings _settings;

        protected override void Configure(IContainerBuilder builder)
        {
            CreatureRaceSettings settings = _settings;
            if (settings == null)
            {
                // Build anyway so the HUD can say what is wrong; the empty settings carry no
                // rig, and the race system reports that as its error.
                Debug.LogError("[CreatureRace] the lifetime scope has no CreatureRaceSettings. Re-run the "
                             + "creature's scene builder.", this);
                settings = ScriptableObject.CreateInstance<CreatureRaceSettings>();
            }
            new CreatureRaceInstaller(settings.Race).Install(builder);
            builder.Register<MujocoCreatureSpawnSystem>(Lifetime.Singleton).AsImplementedInterfaces();
        }
    }
}
