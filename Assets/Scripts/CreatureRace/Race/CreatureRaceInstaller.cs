using MessagePipe;
using VContainer;
using VContainer.Unity;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// The bindings every creature race scene shares, installed by that scene's lifetime
    /// scope, which adds only its creature's <see cref="ICreatureSpawner"/> (architecture.md:
    /// the scope stays the one place binding happens; this only saves repeating the list).
    ///
    ///   Config  CreatureRaceConfig                        from the scene's settings asset
    ///   Model   CreatureRaceModel                         one racer per lane
    ///   System  CreatureRaceSystem (entry point)          series / race / self-test flow
    ///   View    CreatureRaceHudView, CreatureRaceCameraView  authored in the scene
    ///   Pipe    countdown, race finished, series finished
    /// </summary>
    public sealed class CreatureRaceInstaller : IInstaller
    {
        private readonly CreatureRaceConfig _config;

        public CreatureRaceInstaller(CreatureRaceConfig config)
        {
            _config = config;
        }

        public void Install(IContainerBuilder builder)
        {
            builder.RegisterInstance(_config);
            // One racer model per lane of the settings' racer list, fixed for the session.
            builder.Register<CreatureRaceModel>(Lifetime.Singleton).WithParameter(_config.LaneCount);
            builder.RegisterEntryPoint<CreatureRaceSystem>().AsSelf();

            builder.RegisterComponentInHierarchy<CreatureRaceHudView>();
            builder.RegisterComponentInHierarchy<CreatureRaceCameraView>();

            MessagePipeOptions options = builder.RegisterMessagePipe();
            builder.RegisterMessageBroker<CreatureCountdownMessage>(options);
            builder.RegisterMessageBroker<CreatureRaceFinishedMessage>(options);
            builder.RegisterMessageBroker<CreatureSeriesFinishedMessage>(options);
        }
    }
}
