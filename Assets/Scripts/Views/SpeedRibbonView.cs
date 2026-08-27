using PoRacer.Presentation;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Speed ribbon: a trail that fades in behind a racer once it is actually
    /// moving, tinted to that racer's colour.
    ///
    /// It exists to solve a spectator problem rather than a fidelity one. In a wide
    /// pack shot the creatures are small, similarly shaped and all scrabbling, so
    /// who is quick and who is merely thrashing is hard to read. A ribbon whose
    /// length and opacity track ground speed makes the fast racers legible from the
    /// pack camera, and it does it without a HUD element pointing at them.
    ///
    /// Cost control: one TrailRenderer per racer, all sharing one material, and the
    /// emitter switches off entirely below a walking pace so a stalled field draws
    /// nothing. Speed is sampled from the transform rather than from a body, so it
    /// works the same for every morphology.
    /// </summary>
    [DisallowMultipleComponent]
    internal sealed class SpeedRibbonView : MonoBehaviour
    {
        // Below this the racer is scrabbling, not running, and gets no ribbon.
        private const float MIN_SPEED = 1.2f;
        private const float FULL_SPEED = 4f;
        private const float MAX_TRAIL_SECONDS = 0.45f;
        private const float TRAIL_WIDTH = 0.16f;
        private const float FADE_RATE = 4f;
        // Sampling every frame makes the width jitter with articulation-body
        // twitch; a short average is what the eye actually wants to see.
        private const float SPEED_SMOOTHING = 6f;

        private static Material _sharedMaterial;

        private TrailRenderer _trail;
        private Transform _transform;
        private Vector3 _lastPosition;
        private float _smoothedSpeed;
        private float _strength;

        /// <summary>
        /// Colours the ribbon to match the racer. Called by the spawner right after
        /// the component is added, alongside the body tint.
        /// </summary>
        internal void Initialize(Color tint)
        {
            EnsureTrail();
            // Bright at the head, transparent at the tail, in the racer's own hue
            // so a ribbon is attributable to a creature at a glance.
            var gradient = new Gradient();
            gradient.SetKeys(
                new[]
                {
                    new GradientColorKey(Color.Lerp(tint, Color.white, 0.35f), 0f),
                    new GradientColorKey(tint, 1f)
                },
                new[]
                {
                    new GradientAlphaKey(0.75f, 0f),
                    new GradientAlphaKey(0f, 1f)
                });
            _trail.colorGradient = gradient;
        }

        private void Awake()
        {
            _transform = transform;
            _lastPosition = _transform.position;
            EnsureTrail();
        }

        private void Update()
        {
            Vector3 position = _transform.position;
            float deltaTime = Mathf.Max(Time.deltaTime, 0.0001f);
            // Horizontal speed only: a racer bouncing in place is not fast.
            Vector3 travel = position - _lastPosition;
            travel.y = 0f;
            _lastPosition = position;

            float instantSpeed = travel.magnitude / deltaTime;
            _smoothedSpeed = Mathf.Lerp(_smoothedSpeed, instantSpeed, deltaTime * SPEED_SMOOTHING);

            float target = Mathf.InverseLerp(MIN_SPEED, FULL_SPEED, _smoothedSpeed);
            _strength = Mathf.MoveTowards(_strength, target, deltaTime * FADE_RATE);

            if (_strength <= 0.01f)
            {
                if (_trail.emitting)
                {
                    _trail.emitting = false;
                }
                return;
            }
            if (!_trail.emitting)
            {
                _trail.emitting = true;
            }
            _trail.time = MAX_TRAIL_SECONDS * _strength;
            _trail.widthMultiplier = TRAIL_WIDTH * _strength;
        }

        private void EnsureTrail()
        {
            if (_trail != null)
            {
                return;
            }
            _trail = gameObject.AddComponent<TrailRenderer>();
            _trail.emitting = false;
            _trail.time = MAX_TRAIL_SECONDS;
            _trail.widthMultiplier = TRAIL_WIDTH;
            _trail.minVertexDistance = 0.08f;
            _trail.numCapVertices = 2;
            _trail.alignment = LineAlignment.View;
            _trail.textureMode = LineTextureMode.Stretch;
            _trail.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            _trail.receiveShadows = false;
            _trail.sharedMaterial = SharedMaterial();
        }

        /// <summary>
        /// One additive material for every ribbon in the field, so a hundred trails
        /// stay in one batch instead of becoming a hundred material states.
        /// </summary>
        private static Material SharedMaterial()
        {
            if (_sharedMaterial == null)
            {
                _sharedMaterial = FxUtil.GlowParticleMaterial();
            }
            return _sharedMaterial;
        }
    }
}
