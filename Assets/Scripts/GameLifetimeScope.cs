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

            builder.RegisterEntryPoint<Systems_AppBootstrap>();
            builder.RegisterEntryPoint<Systems_Race>().AsSelf();
            builder.RegisterEntryPoint<Systems_Commentary>();
            builder.RegisterEntryPoint<Systems_Spawn>().AsSelf();
            builder.Register<Systems_Persistence>(Lifetime.Singleton);
            builder.Register<Systems_Elo>(Lifetime.Singleton);
            builder.Register<Systems_CameraDirector>(Lifetime.Singleton);
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

            MessagePipeOptions options = builder.RegisterMessagePipe();
            builder.RegisterMessageBroker<RaceStartedMessage>(options);
            builder.RegisterMessageBroker<RacerFinishedMessage>(options);
            builder.RegisterMessageBroker<RacerDnfMessage>(options);
            builder.RegisterMessageBroker<RaceFinishedMessage>(options);

            // Systems_Elo has no tick/start interface; force eager construction so
            // its RaceFinishedMessage subscription exists before the first race ends.
            builder.RegisterBuildCallback(container => container.Resolve<Systems_Elo>());
        }
    }
}
