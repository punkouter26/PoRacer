using PoRacer.Systems;
using UnityEngine;

namespace PoRacer.Models
{
    /// <summary>
    /// One weather and time-of-day look: sun, sky, fog, ambient, grade, the
    /// floating particles and the colour of the ground the footfalls kick up.
    /// Systems_Sky picks one per race from the presets whose <see cref="Kinds"/>
    /// include the track being raced; PostFxView applies it.
    ///
    /// A preset with no kinds listed is a fallback: it serves any track that no
    /// other preset claims.
    ///
    /// Colour rule: the whole frame is tinted by these, so a preset must never push
    /// the image toward saturated red or green. Those two are the racer legend
    /// (heuristic bots and the baseline policy) and have to stay readable.
    /// </summary>
    [CreateAssetMenu(menuName = "PoRacer/Sky Preset")]
    public sealed class SkyPreset : ScriptableObject
    {
        [field: SerializeField] public string DisplayName { get; private set; } = "Clear";
        [field: SerializeField] public TrackKind[] Kinds { get; private set; } = System.Array.Empty<TrackKind>();

        [field: Header("Sun")]
        [field: SerializeField] public float SunIntensity { get; private set; } = 1.35f;
        [field: SerializeField] public float SunTemperature { get; private set; } = 5600f;
        [field: SerializeField] public float SunPitch { get; private set; } = 42f;
        [field: SerializeField] public float SunYaw { get; private set; } = -35f;

        [field: Header("Sky and fog")]
        [field: SerializeField] public Color SkyTint { get; private set; } = new(0.45f, 0.65f, 0.95f);
        [field: SerializeField] public float Atmosphere { get; private set; } = 0.9f;
        [field: SerializeField] public float SkyExposure { get; private set; } = 1.25f;
        [field: SerializeField] public Color FogColor { get; private set; } = new(0.62f, 0.7f, 0.82f);
        [field: SerializeField] public float FogStart { get; private set; } = 45f;
        [field: SerializeField] public float FogEnd { get; private set; } = 140f;

        [field: Header("Ambient")]
        [field: SerializeField] public Color AmbientSky { get; private set; } = new(0.55f, 0.62f, 0.75f);
        [field: SerializeField] public Color AmbientEquator { get; private set; } = new(0.42f, 0.42f, 0.45f);
        [field: SerializeField] public Color AmbientGround { get; private set; } = new(0.22f, 0.2f, 0.19f);

        [field: Header("Grade")]
        [field: SerializeField] public Color GradeFilter { get; private set; } = Color.white;
        [field: SerializeField] public float Saturation { get; private set; } = 10f;
        [field: SerializeField] public float Vignette { get; private set; } = 0.22f;
        [field: SerializeField, Range(0f, 1f)] public float Wetness { get; private set; }

        [field: Header("Air particles")]
        [field: SerializeField] public Color AirColor { get; private set; } = new(1f, 1f, 0.9f, 0.3f);
        [field: SerializeField] public float AirRate { get; private set; } = 10f;
        [field: SerializeField] public float AirSizeMin { get; private set; } = 0.03f;
        [field: SerializeField] public float AirSizeMax { get; private set; } = 0.08f;
        [field: SerializeField] public float AirRise { get; private set; } = -0.1f;
        [field: SerializeField] public float AirWindX { get; private set; } = 0.6f;
        [field: SerializeField] public float AirWander { get; private set; } = 0.3f;
        /// <summary>Falling streaks (drizzle) instead of drifting motes.</summary>
        [field: SerializeField] public bool AirStreaks { get; private set; }

        [field: Header("Ground")]
        /// <summary>Colour of footprints and the dust a footfall kicks up.</summary>
        [field: SerializeField] public Color GroundTone { get; private set; } = new(0.62f, 0.55f, 0.45f);
    }
}
