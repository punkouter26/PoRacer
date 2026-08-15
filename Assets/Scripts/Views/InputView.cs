using PoRacer.Systems;
using UnityEngine;
using UnityEngine.InputSystem;
using VContainer;

namespace PoRacer.Views
{
    /// <summary>
    /// The only class that touches PlayerControls. Camera map only:
    /// Next/Prev cycle chase targets, Overview returns to the wide shot.
    /// </summary>
    public sealed class InputView : MonoBehaviour
    {
        private PlayerControls _controls;
        private Systems_CameraDirector _cameraDirector;

        [Inject]
        public void Construct(Systems_CameraDirector cameraDirector)
        {
            _cameraDirector = cameraDirector;
        }

        private void Awake()
        {
            _controls = new PlayerControls();
        }

        private void OnEnable()
        {
            _controls.Camera.Enable();
            _controls.Camera.Next.performed += OnNext;
            _controls.Camera.Prev.performed += OnPrev;
            _controls.Camera.Overview.performed += OnOverview;
        }

        private void OnDisable()
        {
            _controls.Camera.Next.performed -= OnNext;
            _controls.Camera.Prev.performed -= OnPrev;
            _controls.Camera.Overview.performed -= OnOverview;
            _controls.Camera.Disable();
        }

        private void OnNext(InputAction.CallbackContext context) => _cameraDirector.NextTarget();

        private void OnPrev(InputAction.CallbackContext context) => _cameraDirector.PrevTarget();

        private void OnOverview(InputAction.CallbackContext context) => _cameraDirector.ShowOverview();
    }
}
