using Unity.Cinemachine;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Scene anchor for the Cinemachine cameras. Pure data exposure — no logic.
    /// </summary>
    public sealed class CameraRigView : MonoBehaviour
    {
        [SerializeField] private CinemachineCamera _overviewCamera;

        public CinemachineCamera OverviewCamera => _overviewCamera;
    }
}
