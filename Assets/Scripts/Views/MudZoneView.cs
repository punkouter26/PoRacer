using PoRacer.Presentation;
using System.Collections.Generic;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Hazard trigger volume built by Systems_TrackBuilder: any ArticulationBody
    /// part inside gets a viscous drag force, so creatures wade slowly through
    /// mud. Bodies are collected in OnTriggerStay and forced once per
    /// FixedUpdate so multi-collider limbs are never double-damped.
    /// Entering the pit splashes mud particles. The looping squelch that used to
    /// rise with the number of bodies wading went with the rest of the
    /// non-creature mix; the particles and their material are built in code.
    /// </summary>
    [RequireComponent(typeof(BoxCollider))]
    public sealed class MudZoneView : MonoBehaviour
    {
        // Deceleration per m/s of speed; creatures crawl at ~0.5-2 m/s.
        private const float MUD_DRAG_PER_SPEED = 2f;
        private const float MIN_SPLASH_SPEED = 0.3f;
        private const int SPLASH_PARTICLES = 10;

        private static readonly Color MudColor = new(0.4f, 0.28f, 0.13f);

        private readonly List<ArticulationBody> _bodiesInMud = new();
        private ParticleSystem _splash;
        private ParticleSystem.EmitParams _emitParams;

        private void Awake()
        {
            _splash = BuildSplash();
        }

        private void OnTriggerEnter(Collider other)
        {
            ArticulationBody body = other.attachedArticulationBody;
            if (body == null || body.linearVelocity.magnitude < MIN_SPLASH_SPEED)
            {
                return;
            }
            _emitParams.position = other.bounds.center;
            _splash.Emit(_emitParams, SPLASH_PARTICLES);
        }

        private void OnTriggerStay(Collider other)
        {
            ArticulationBody body = other.attachedArticulationBody;
            if (body != null && !_bodiesInMud.Contains(body))
            {
                _bodiesInMud.Add(body);
            }
        }

        private void FixedUpdate()
        {
            for (int bodyIndex = 0; bodyIndex < _bodiesInMud.Count; bodyIndex++)
            {
                ArticulationBody body = _bodiesInMud[bodyIndex];
                if (body != null)
                {
                    body.AddForce(-body.linearVelocity * (MUD_DRAG_PER_SPEED * body.mass));
                }
            }
            _bodiesInMud.Clear();
        }

        private ParticleSystem BuildSplash()
        {
            var go = new GameObject("MudSplash");
            go.transform.SetParent(transform, false);
            var ps = go.AddComponent<ParticleSystem>();

            ParticleSystem.MainModule main = ps.main;
            main.playOnAwake = false;
            main.simulationSpace = ParticleSystemSimulationSpace.World;
            main.startLifetime = new ParticleSystem.MinMaxCurve(0.4f, 0.8f);
            main.startSpeed = new ParticleSystem.MinMaxCurve(1.2f, 3f);
            main.startSize = new ParticleSystem.MinMaxCurve(0.06f, 0.18f);
            main.startColor = new ParticleSystem.MinMaxGradient(
                MudColor,
                new Color(MudColor.r * 0.7f, MudColor.g * 0.7f, MudColor.b * 0.7f));
            main.gravityModifier = 1.4f;
            main.maxParticles = 200;

            // A slow ambient blorp telegraphs the pit as a hazard even when nothing
            // is wading; entry splashes still come from Emit().
            ParticleSystem.EmissionModule emission = ps.emission;
            emission.rateOverTime = 2.5f;

            ParticleSystem.ShapeModule shape = ps.shape;
            shape.shapeType = ParticleSystemShapeType.Cone;
            shape.angle = 35f;
            shape.radius = 0.15f;
            shape.rotation = new Vector3(-90f, 0f, 0f);

            var splashRenderer = ps.GetComponent<ParticleSystemRenderer>();
            // Shared soft sprite so droplets look like goop, not squares.
            Material soft = FxUtil.SoftParticleMaterial();
            if (soft != null)
            {
                splashRenderer.material = soft;
            }
            splashRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            splashRenderer.receiveShadows = false;
            return ps;
        }

    }
}
