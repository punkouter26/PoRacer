using System;
using MessagePipe;
using PoRacer.Models;
using PoRacer.Presentation;
using UnityEngine;
using UnityEngine.UIElements;
using VContainer;

namespace PoRacer.Views
{
    /// <summary>
    /// Broadcast lower-third: says why the director cut ("DUEL  Crab #1 vs Quad #2",
    /// "GETTING UP  Isaac H1 #1"). Built once on the HUD's document; a cut only sets one
    /// string and restarts one fade, so it allocates only when the shot changes.
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    public sealed class ShotCaptionView : MonoBehaviour
    {
        private const float CAPTION_TOP_PERCENT = 18f;
        private const int CAPTION_SHOW_MS = 3200;
        private const int CAPTION_FADE_MS = 300;
        private const float CAPTION_SLIDE_PX = 60f;
        private const float MARKER_WIDTH = 4f;

        private static readonly Action<VisualElement, float> CaptionTick = (element, value) =>
        {
            float elapsedMs = value * CAPTION_SHOW_MS;
            float appear;
            if (elapsedMs < CAPTION_FADE_MS)
            {
                float remaining = 1f - elapsedMs / CAPTION_FADE_MS;
                appear = 1f - remaining * remaining;
            }
            else if (elapsedMs < CAPTION_SHOW_MS - CAPTION_FADE_MS)
            {
                appear = 1f;
            }
            else
            {
                appear = (CAPTION_SHOW_MS - elapsedMs) / CAPTION_FADE_MS;
            }
            element.style.opacity = appear;
            element.style.translate = new Translate(-CAPTION_SLIDE_PX * (1f - appear), 0f);
            if (value >= 1f)
            {
                element.style.display = DisplayStyle.None;
            }
        };

        private RaceModel _raceModel;
        private IDisposable _subscription;
        private VisualElement _caption;
        private VisualElement _marker;
        private Label _kindLabel;
        private Label _subjectLabel;

        [Inject]
        public void Construct(RaceModel raceModel, ISubscriber<CameraShotChangedMessage> shotChanged)
        {
            _raceModel = raceModel;
            _subscription = shotChanged.Subscribe(OnShotChanged);
        }

        private void Start()
        {
            VisualElement root = GetComponent<UIDocument>().rootVisualElement;
            VisualElement safeRoot = UiTheme.BuildSafeRoot(root);

            _caption = new VisualElement { pickingMode = PickingMode.Ignore };
            _caption.style.position = Position.Absolute;
            _caption.style.top = new Length(CAPTION_TOP_PERCENT, LengthUnit.Percent);
            _caption.style.left = UiTheme.SPACE_MD;
            _caption.style.maxWidth = new Length(80f, LengthUnit.Percent);
            _caption.style.flexDirection = FlexDirection.Row;
            _caption.style.alignItems = Align.Center;
            UiTheme.StyleGlassPanel(_caption);
            _caption.style.display = DisplayStyle.None;
            safeRoot.Add(_caption);

            _marker = new VisualElement { pickingMode = PickingMode.Ignore };
            _marker.style.width = MARKER_WIDTH;
            _marker.style.alignSelf = Align.Stretch;
            _marker.style.flexShrink = 0f;
            _marker.style.backgroundColor = UiTheme.Accent;
            _marker.style.marginRight = UiTheme.SPACE_SM;
            _caption.Add(_marker);

            _kindLabel = new Label { pickingMode = PickingMode.Ignore };
            _kindLabel.style.color = UiTheme.Gold;
            _kindLabel.style.fontSize = UiTheme.FONT_SM;
            _kindLabel.style.unityFontStyleAndWeight = FontStyle.Bold;
            _kindLabel.style.letterSpacing = 1.5f;
            _kindLabel.style.marginRight = UiTheme.SPACE_SM;
            UiTheme.ApplyFont(_kindLabel);
            _caption.Add(_kindLabel);

            _subjectLabel = new Label { pickingMode = PickingMode.Ignore };
            _subjectLabel.style.color = UiTheme.Text;
            _subjectLabel.style.fontSize = UiTheme.FONT_SM;
            _subjectLabel.style.flexShrink = 1f;
            _subjectLabel.style.whiteSpace = WhiteSpace.Normal;
            UiTheme.ApplyFont(_subjectLabel);
            _caption.Add(_subjectLabel);
        }

        private void OnDestroy() => _subscription?.Dispose();

        private void OnShotChanged(CameraShotChangedMessage message)
        {
            if (_caption == null)
            {
                return;
            }
            string kind = KindText(message.Reason);
            RacerState subject = message.SubjectId != null ? _raceModel.FindRacer(message.SubjectId) : null;
            if (kind == null || subject == null)
            {
                _caption.style.display = DisplayStyle.None;
                return;
            }
            RacerState rival = message.RivalId != null ? _raceModel.FindRacer(message.RivalId) : null;
            _kindLabel.text = kind;
            _subjectLabel.text = rival != null
                ? $"{subject.DisplayName}  vs  {rival.DisplayName}"
                : subject.DisplayName;
            _marker.style.backgroundColor = TrainerTeams.ColorOf(subject.TrainedBy);
            _caption.style.opacity = 0f;
            _caption.style.display = DisplayStyle.Flex;
            _caption.experimental.animation.Start(0f, 1f, CAPTION_SHOW_MS, CaptionTick);
        }

        /// <summary>The caption headline for a cut, or null for shots that speak for themselves.</summary>
        private static string KindText(ShotReason reason)
        {
            switch (reason)
            {
                case ShotReason.Leader:
                    return "LEADER";
                case ShotReason.NewLeader:
                    return "NEW LEADER";
                case ShotReason.Duel:
                    return "DUEL";
                case ShotReason.GoingDown:
                    return "GOING DOWN";
                case ShotReason.GettingUp:
                    return "GETTING UP";
                case ShotReason.BackUp:
                    return "BACK ON ITS FEET";
                case ShotReason.FinalMetres:
                    return "FINAL METRES";
                case ShotReason.ViewerPick:
                    return "YOUR PICK";
                default:
                    return null;
            }
        }
    }
}
