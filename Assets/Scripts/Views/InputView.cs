using PoRacer.Models;
using PoRacer.Systems;
using UnityEngine;
using UnityEngine.InputSystem;
using VContainer;

namespace PoRacer.Views
{
    /// <summary>
    /// The only class that touches PlayerControls. Camera map only:
    /// Next/Prev cycle chase targets, Overview returns to the wide shot.
    ///
    /// Touch. The keyboard bindings are the desktop half; on a phone there was no
    /// way to change camera at all, which is most of the point of watching a race.
    /// A swipe cannot be expressed as a binding, so Point/Press carry the raw
    /// pointer and the gesture is recognised here: horizontal swipe = next/prev
    /// racer, tap = overview. That is still View work — it reads input and calls a
    /// System, and Systems_CameraDirector never learns where the call came from.
    /// &lt;Pointer&gt; covers touchscreen and mouse alike, so a desktop click-drag
    /// cycles racers too.
    /// </summary>
    public sealed class InputView : MonoBehaviour
    {
        // Gesture thresholds as a fraction of screen width, not pixels: the same
        // finger movement is a very different pixel count on a 1080p phone and a
        // 4K desktop.
        private const float SWIPE_MIN_WIDTH_FRACTION = 0.12f;
        private const float TAP_MAX_WIDTH_FRACTION = 0.03f;
        private const float TAP_MAX_SECONDS = 0.4f;

        private PlayerControls _controls;
        private Systems_CameraDirector _cameraDirector;
        private RaceConfigModel _config;
        private Vector2 _pressPosition;
        private float _pressTime;
        private bool _pressed;

        [Inject]
        public void Construct(Systems_CameraDirector cameraDirector, RaceConfigModel config)
        {
            _cameraDirector = cameraDirector;
            _config = config;
        }

        private void Awake()
        {
            _controls ??= new PlayerControls();
        }

        private void OnEnable()
        {
            // A domain reload while playing wipes this non-serialized field and
            // re-runs OnEnable without Awake, so it has to be able to rebuild.
            _controls ??= new PlayerControls();
            _controls.Camera.Enable();
            _controls.Camera.Next.performed += OnNext;
            _controls.Camera.Prev.performed += OnPrev;
            _controls.Camera.Overview.performed += OnOverview;
            _controls.Camera.Press.started += OnPressStarted;
            _controls.Camera.Press.canceled += OnPressEnded;
        }

        private void OnDisable()
        {
            if (_controls == null)
            {
                return;
            }
            _controls.Camera.Next.performed -= OnNext;
            _controls.Camera.Prev.performed -= OnPrev;
            _controls.Camera.Overview.performed -= OnOverview;
            _controls.Camera.Press.started -= OnPressStarted;
            _controls.Camera.Press.canceled -= OnPressEnded;
            _controls.Camera.Disable();
            _pressed = false;
        }

        private void OnNext(InputAction.CallbackContext context) => _cameraDirector.NextTarget();

        private void OnPrev(InputAction.CallbackContext context) => _cameraDirector.PrevTarget();

        private void OnOverview(InputAction.CallbackContext context) => _cameraDirector.ShowOverview();

        private void OnPressStarted(InputAction.CallbackContext context)
        {
            _pressed = true;
            _pressPosition = _controls.Camera.Point.ReadValue<Vector2>();
            // Unscaled: the winner slow-mo must not stretch the tap window.
            _pressTime = Time.unscaledTime;
        }

        /// <summary>
        /// Classifies the finished drag. Menu taps are ignored outright: the menu
        /// sits over the race on its own panel, and its buttons are live touch
        /// targets that should not also be steering a camera behind them.
        /// </summary>
        private void OnPressEnded(InputAction.CallbackContext context)
        {
            if (!_pressed)
            {
                return;
            }
            _pressed = false;
            if (_config != null && _config.MenuVisible)
            {
                return;
            }
            Vector2 travel = _controls.Camera.Point.ReadValue<Vector2>() - _pressPosition;
            float width = Mathf.Max(1f, Screen.width);
            float swipeMin = width * SWIPE_MIN_WIDTH_FRACTION;
            float tapMax = width * TAP_MAX_WIDTH_FRACTION;

            if (Mathf.Abs(travel.x) >= swipeMin && Mathf.Abs(travel.x) > Mathf.Abs(travel.y))
            {
                // Swipe left pulls the next racer in from the right, the way a
                // carousel moves under the finger.
                if (travel.x < 0f)
                {
                    _cameraDirector.NextTarget();
                }
                else
                {
                    _cameraDirector.PrevTarget();
                }
                return;
            }
            if (travel.magnitude <= tapMax && Time.unscaledTime - _pressTime <= TAP_MAX_SECONDS)
            {
                _cameraDirector.ShowOverview();
            }
        }
    }
}
