using PoRacer.CreatureRace;
using UnityEngine;
using VContainer;
using VContainer.Unity;

namespace PoRacer.WormRace
{
    /// <summary>
    /// Composition root of SCN_WORM_RACE: the creature template's shared race bindings
    /// (CreatureRaceInstaller: config, model, race system, HUD, camera, MessagePipe) plus the
    /// worm's own spawner, which adds the PhysX lane and the cross-simulator stand-ins.
    /// Every class asks for exactly what it uses (architecture.md: no GameContext).
    ///
    ///   Settings  WormRaceSettings (its Race config + the PhysX knobs)
    ///   Spawner   WormSpawnSystem : ICreatureSpawner   MuJoCo worms (generic builder), PhysX worms, stand-ins
    ///   View      MujocoCreatureView, PhysxWormView     spawned per race, bound by the builders
    /// </summary>
    public sealed class WormRaceLifetimeScope : LifetimeScope
    {
        [Tooltip("Assets/WormRace/WormRaceSettings.asset.")]
        [SerializeField] private WormRaceSettings _settings;

        protected override void Configure(IContainerBuilder builder)
        {
            WormRaceSettings settings = _settings;
            if (settings == null)
            {
                // Build anyway so the HUD can say what is wrong; the empty settings carry
                // no rig, and the race system reports that as its error.
                Debug.LogError("[WormRace] WormRaceLifetimeScope has no WormRaceSettings. "
                             + "Assign " + WormRacePaths.SETTINGS + " in the Inspector.", this);
                settings = ScriptableObject.CreateInstance<WormRaceSettings>();
            }
            builder.RegisterInstance(settings);
            new CreatureRaceInstaller(settings.Race).Install(builder);
            builder.Register<WormSpawnSystem>(Lifetime.Singleton).AsImplementedInterfaces();
        }
    }
}
