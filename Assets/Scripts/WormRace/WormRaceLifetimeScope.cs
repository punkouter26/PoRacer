using MessagePipe;
using UnityEngine;
using VContainer;
using VContainer.Unity;

namespace PoRacer.WormRace
{
    /// <summary>
    /// Composition root of SCN_WORM_RACE. The only place anything in the worm race is
    /// bound; every class asks for exactly what it uses (architecture.md: no GameContext).
    ///
    ///   Model   WormRaceModel                          race state, one racer per settings lane
    ///   System  WormRaceSystem (entry point)           series / race / self-test flow
    ///           WormSpawnSystem                        worms, MuJoCo world, stand-ins
    ///   View    WormRaceHudView, WormRaceCameraView    authored in the scene
    ///           MujocoWormView, PhysxWormView          spawned per race, bound by the builders
    ///   Pipe    countdown, race finished, series finished
    /// </summary>
    public sealed class WormRaceLifetimeScope : LifetimeScope
    {
        [Tooltip("Assets/WormRace/WormRaceSettings.asset, wired by Editor_BuildWormRaceScene.")]
        [SerializeField] private WormRaceSettings _settings;

        protected override void Configure(IContainerBuilder builder)
        {
            WormRaceSettings settings = _settings;
            if (settings == null)
            {
                // Build anyway so the HUD can say what is wrong; the empty settings carry
                // no rig, and the race system reports that as its error.
                Debug.LogError("[WormRace] WormRaceLifetimeScope has no WormRaceSettings. "
                             + "Run PoRacer.WormRace.EditorTools.Editor_BuildWormRaceScene.Build().", this);
                settings = ScriptableObject.CreateInstance<WormRaceSettings>();
            }
            builder.RegisterInstance(settings);

            // One racer model per lane of the settings' racer list, fixed for the session.
            builder.Register<WormRaceModel>(Lifetime.Singleton).WithParameter(settings.LaneCount);
            builder.Register<WormSpawnSystem>(Lifetime.Singleton);
            builder.RegisterEntryPoint<WormRaceSystem>().AsSelf();

            builder.RegisterComponentInHierarchy<WormRaceHudView>();
            builder.RegisterComponentInHierarchy<WormRaceCameraView>();

            MessagePipeOptions options = builder.RegisterMessagePipe();
            builder.RegisterMessageBroker<WormCountdownMessage>(options);
            builder.RegisterMessageBroker<WormRaceFinishedMessage>(options);
            builder.RegisterMessageBroker<WormSeriesFinishedMessage>(options);
        }
    }
}
