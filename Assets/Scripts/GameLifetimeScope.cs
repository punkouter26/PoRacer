using MessagePipe;
using PoRacer.Models;
using PoRacer.Systems;
using PoRacer.Views;
using UnityEngine;
using VContainer;
using VContainer.Unity;

namespace PoRacer
{
    public sealed class GameLifetimeScope : LifetimeScope
    {
        [SerializeField] private CreatureCatalog _catalog;
        [SerializeField] private FruitCatalog _fruitCatalog;
        [SerializeField] private SkyCatalog _skyCatalog;

        protected override void Configure(IContainerBuilder builder)
        {
            builder.RegisterInstance(_catalog);
            builder.RegisterInstance(_fruitCatalog);
            builder.RegisterInstance(_skyCatalog);

            builder.Register<RaceModel>(Lifetime.Singleton);
            builder.Register<EloModel>(Lifetime.Singleton);
            builder.Register<RaceConfigModel>(Lifetime.Singleton);
            builder.Register<AudioMixModel>(Lifetime.Singleton);
            builder.Register<RaceTelemetryModel>(Lifetime.Singleton);
            builder.Register<SimWarsModel>(Lifetime.Singleton);
            builder.Register<QualityModel>(Lifetime.Singleton);
            builder.Register<SkyModel>(Lifetime.Singleton);
            builder.Register<HapticsModel>(Lifetime.Singleton);

            builder.RegisterEntryPoint<Systems_AppBootstrap>();
            // Entry point so it starts warming the moment the menu appears, which is the
            // whole point: the first race used to stall ~4.4 s instantiating the grid
            // when START was pressed. AsSelf as well, because Systems_Spawn waits on
            // its IsComplete before it builds one.
            builder.RegisterEntryPoint<Systems_Warmup>().AsSelf();
            builder.RegisterEntryPoint<Systems_Race>().AsSelf();
            builder.RegisterEntryPoint<Systems_LeadWatcher>();
            // Entry point: its Tick runs the duck release and the menu mix slide.
            builder.RegisterEntryPoint<Systems_AudioMix>().AsSelf();
            builder.RegisterEntryPoint<Systems_Spawn>().AsSelf();
            // Entry point so its RaceFinishedMessage subscription exists before the first race ends.
            builder.RegisterEntryPoint<Systems_FruitPour>().AsSelf();
            builder.Register<Systems_Persistence>(Lifetime.Singleton);
            builder.Register<Systems_Elo>(Lifetime.Singleton);
            builder.Register<Systems_SimWars>(Lifetime.Singleton);
            // Entry point: FixedTick integrates joint power per physics step, Tick reads
            // pose and policy outputs per frame. AsSelf so the spawner can register racers.
            builder.RegisterEntryPoint<Systems_RacerTelemetry>().AsSelf();
            // Entry point: its Tick watches for the final stretch to re-aim the shot.
            builder.RegisterEntryPoint<Systems_CameraDirector>().AsSelf();
            // Entry point: its Tick measures race frames to pick the quality tier.
            builder.RegisterEntryPoint<Systems_QualityGovernor>();
            // Entry point: its Start shows the selected map's sky behind the menu.
            builder.RegisterEntryPoint<Systems_Sky>();
            // Entry point: its Start loads the vibration setting. AsSelf for the menu toggle.
            builder.RegisterEntryPoint<Systems_Haptics>().AsSelf();
            builder.Register(container =>
            {
                var track = container.Resolve<Views.RaceTrackView>();
                return new Systems_TrackBuilder(track.GroundMaterial, track.ObstacleMaterial, track.PhysicsMaterial);
            }, Lifetime.Singleton);

            builder.RegisterComponentInHierarchy<RaceHudView>();
            builder.RegisterComponentInHierarchy<MenuView>();
            builder.RegisterComponentInHierarchy<WinFxView>();
            builder.RegisterComponentInHierarchy<InputView>();
            builder.RegisterComponentInHierarchy<RaceTrackView>();
            builder.RegisterComponentInHierarchy<CameraRigView>();
            builder.RegisterComponentInHierarchy<FinishLineView>();
            builder.RegisterComponentInHierarchy<CameraFxView>();
            builder.RegisterComponentInHierarchy<DebugOverlayView>();
            builder.RegisterComponentInHierarchy<PostFxView>();
            builder.RegisterComponentInHierarchy<TelemetryCardView>();
            builder.RegisterComponentInHierarchy<ShotCaptionView>();
            builder.RegisterComponentInHierarchy<SimWarsView>();
            builder.RegisterComponentInHierarchy<HapticsView>();

            MessagePipeOptions options = builder.RegisterMessagePipe();
            builder.RegisterMessageBroker<RaceStartedMessage>(options);
            builder.RegisterMessageBroker<RacerFinishedMessage>(options);
            builder.RegisterMessageBroker<RacerDnfMessage>(options);
            builder.RegisterMessageBroker<LeadChangedMessage>(options);
            builder.RegisterMessageBroker<RaceFinishedMessage>(options);
            // These three were declared and published but never registered, so
            // VContainer handed every publisher and subscriber the null default
            // its optional parameter allowed: wipeout audio, the wipeout camera
            // shake, the photo-finish sting and its commentary line were all dead.
            builder.RegisterMessageBroker<RacerWipeoutMessage>(options);
            builder.RegisterMessageBroker<RacerOvertakeMessage>(options);
            builder.RegisterMessageBroker<PhotoFinishMessage>(options);
            builder.RegisterMessageBroker<CameraShotChangedMessage>(options);
            builder.RegisterMessageBroker<TrackBuiltMessage>(options);

            // Systems_Elo has no tick/start interface; force eager construction so
            // its RaceFinishedMessage subscription exists before the first race ends.
            builder.RegisterBuildCallback(container => container.Resolve<Systems_Elo>());
            // Same for the league: it must be listening before the first race ends.
            builder.RegisterBuildCallback(container => container.Resolve<Systems_SimWars>());
        }
    }
}
