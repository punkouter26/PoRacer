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

        protected override void Configure(IContainerBuilder builder)
        {
            builder.RegisterInstance(_catalog);

            builder.Register<RaceModel>(Lifetime.Singleton);
            builder.Register<EloModel>(Lifetime.Singleton);
            builder.Register<RaceConfigModel>(Lifetime.Singleton);
            builder.Register<CommentaryModel>(Lifetime.Singleton);
            builder.Register<AudioMixModel>(Lifetime.Singleton);

            builder.RegisterEntryPoint<Systems_AppBootstrap>();
            builder.RegisterEntryPoint<Systems_Race>().AsSelf();
            builder.RegisterEntryPoint<Systems_Commentary>();
            // Entry point: its Tick runs the duck release and the menu mix slide.
            builder.RegisterEntryPoint<Systems_AudioMix>().AsSelf();
            builder.RegisterEntryPoint<Systems_Spawn>().AsSelf();
            builder.Register<Systems_Persistence>(Lifetime.Singleton);
            builder.Register<Systems_Elo>(Lifetime.Singleton);
            // Entry point: its Tick watches for the final stretch to re-aim the shot.
            builder.RegisterEntryPoint<Systems_CameraDirector>().AsSelf();
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
            builder.RegisterComponentInHierarchy<AudioDirectorView>();
            builder.RegisterComponentInHierarchy<CameraFxView>();
            builder.RegisterComponentInHierarchy<DebugOverlayView>();
            builder.RegisterComponentInHierarchy<PostFxView>();

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

            // Systems_Elo has no tick/start interface; force eager construction so
            // its RaceFinishedMessage subscription exists before the first race ends.
            builder.RegisterBuildCallback(container => container.Resolve<Systems_Elo>());
        }
    }
}
