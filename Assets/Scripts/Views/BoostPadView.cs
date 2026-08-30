using PoRacer.Presentation;
using System.Collections.Generic;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Hazard trigger volume built by Systems_TrackBuilder: any ArticulationBody
    /// part inside gets a forward push, so pads reward racers whose line crosses
    /// them. Bodies are collected in OnTriggerStay and forced once per
    /// FixedUpdate so multi-collider limbs are never double-boosted.
    /// Entering the pad sparks particles. It used to whoosh too; that went with the
    /// rest of the non-creature mix, so the pad is now seen and not heard.
    /// </summary>
    [RequireComponent(typeof(BoxCollider))]
    public sealed class BoostPadView : MonoBehaviour
    {
        // Forward acceleration while inside; creatures crawl at ~0.5-2 m/s, so a
        // pad crossing gives a visible surge without launching anyone.
        private const float BOOST_ACCEL = 5f;
        private const int SPARK_PARTICLES = 14;

        private static readonly Color BoostColor = new(0.3f, 1f, 0.6f);

        private readonly List<ArticulationBody> _bodiesInPad = new();
        private ParticleSystem _sparks;
        private ParticleSystem.EmitParams _emitParams;

        private void Awake()
        {
            _sparks = BuildSparks();
        }

        private void OnTriggerEnter(Collider other)
        {
            if (other.attachedArticulationBody == null)
            {
                return;
            }
            _emitParams.position = other.bounds.center;
            _sparks.Emit(_emitParams, SPARK_PARTICLES);
        }

        private void OnTriggerStay(Collider other)
        {
            ArticulationBody body = other.attachedArticulationBody;
            if (body != null && !_bodiesInPad.Contains(body))
            {
                _bodiesInPad.Add(body);
            }
        }

        private void FixedUpdate()
        {
            for (int bodyIndex = 0; bodyIndex < _bodiesInPad.Count; bodyIndex++)
            {
                ArticulationBody body = _bodiesInPad[bodyIndex];
                if (body != null)
                {
                    body.AddForce(Vector3.forward * (BOOST_ACCEL * body.mass));
                }
            }
            _bodiesInPad.Clear();
        }

        private ParticleSystem BuildSparks()
        {
            var go = new GameObject("BoostSparks");
            go.transform.SetParent(transform, false);
            var ps = go.AddComponent<ParticleSystem>();

            ParticleSystem.MainModule main = ps.main;
            main.playOnAwake = false;
            main.simulationSpace = ParticleSystemSimulationSpace.World;
            main.startLifetime = new ParticleSystem.MinMaxCurve(0.3f, 0.6f);
            main.startSpeed = new ParticleSystem.MinMaxCurve(2f, 4.5f);
            main.startSize = new ParticleSystem.MinMaxCurve(0.05f, 0.14f);
            main.startColor = new ParticleSystem.MinMaxGradient(
                BoostColor,
                new Color(0.7f, 1f, 0.85f));
            main.gravityModifier = 0.4f;
            main.maxParticles = 200;

            // A slow ambient sparkle telegraphs the pad as a bonus even when
            // nothing is crossing; entry bursts still come from Emit().
            ParticleSystem.EmissionModule emission = ps.emission;
            emission.rateOverTime = 4f;

            ParticleSystem.ShapeModule shape = ps.shape;
            shape.shapeType = ParticleSystemShapeType.Cone;
            shape.angle = 20f;
            shape.radius = 0.4f;
            shape.rotation = new Vector3(-90f, 0f, 0f);

            var sparkRenderer = ps.GetComponent<ParticleSystemRenderer>();
            // Additive glow: overlapping sparks burn hot instead of stacking flat.
            Material glow = FxUtil.GlowParticleMaterial();
            if (glow != null)
            {
                sparkRenderer.material = glow;
            }
            sparkRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            sparkRenderer.receiveShadows = false;
            return ps;
        }

    }
}
