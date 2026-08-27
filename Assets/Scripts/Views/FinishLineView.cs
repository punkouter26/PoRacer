using PoRacer.Systems;
using UnityEngine;
using VContainer;

namespace PoRacer.Views
{
    /// <summary>
    /// Trigger volume across the finish line. Forwards crossings to Systems_Race.
    /// </summary>
    [RequireComponent(typeof(BoxCollider))]
    public sealed class FinishLineView : MonoBehaviour
    {
        private Systems_Race _race;
        private LineRenderer _laser;
        private Light _finishGlow;

        [Inject]
        public void Construct(Systems_Race race)
        {
            _race = race;
        }

        private void Start()
        {
            BuildLaserGate();
        }

        private void Update()
        {
            if (_finishGlow != null)
            {
                _finishGlow.intensity = 1.8f + 0.6f * Mathf.Sin(Time.time * 4f);
            }
        }

        private void BuildLaserGate()
        {
            GameObject laserObj = new GameObject("HolographicLaser");
            laserObj.transform.SetParent(transform, false);

            _laser = laserObj.AddComponent<LineRenderer>();
            _laser.positionCount = 2;
            _laser.SetPosition(0, new Vector3(-12f, 0.15f, 0f));
            _laser.SetPosition(1, new Vector3(12f, 0.15f, 0f));
            _laser.startWidth = 0.18f;
            _laser.endWidth = 0.18f;

            Material glowMat = FxUtil.GlowParticleMaterial();
            if (glowMat != null)
            {
                _laser.material = glowMat;
            }
            _laser.startColor = new Color(0.1f, 0.9f, 1f, 0.85f);
            _laser.endColor = new Color(0.9f, 0.2f, 1f, 0.85f);
            _laser.useWorldSpace = false;

            GameObject lightObj = new GameObject("FinishGlow");
            lightObj.transform.SetParent(transform, false);
            lightObj.transform.localPosition = new Vector3(0f, 1.2f, 0f);
            _finishGlow = lightObj.AddComponent<Light>();
            _finishGlow.type = LightType.Point;
            _finishGlow.range = 14f;
            _finishGlow.color = new Color(0.2f, 0.8f, 1f);
            _finishGlow.intensity = 2f;
        }

        private void OnTriggerEnter(Collider other)
        {
            if (_race == null)
            {
                return;
            }
            RacerView racer = other.GetComponentInParent<RacerView>();
            if (racer != null)
            {
                float overshoot = racer.transform.position.z - transform.position.z;
                _race.NotifyFinish(racer.RacerId, overshoot);
            }
        }
    }
}
