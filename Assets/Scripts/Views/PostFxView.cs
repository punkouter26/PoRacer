using MessagePipe;
using PoRacer.Models;
using PoRacer.Systems;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using VContainer;

namespace PoRacer.Views
{
    /// <summary>
    /// Builds the scene look in code at startup: post-processing volume (bloom,
    /// ACES tonemapping, vignette, chromatic aberration, color grade), procedural
    /// skybox, tri-light ambient, sun + fill light rig, distance fog, and per-map
    /// ambient particles (fireflies, dust, pollen). No asset files needed.
    /// Re-grades the whole mood per selected map: Flat = clean day, Lumpy = warm
    /// dusk, Swamp = murky green. Race start fires an exposure flash pulse.
    /// </summary>
    public sealed class PostFxView : MonoBehaviour
    {
        private const float BASE_EXPOSURE = 0.15f;
        private const float START_FLASH = 0.85f;
        private const float FLASH_DECAY_PER_SECOND = 2.2f;
        private const float SHADOW_DISTANCE = 70f;

        private VolumeProfile _profile;
        private Material _skyboxMaterial;
        private Vignette _vignette;
        private ColorAdjustments _colorAdjustments;
        private Light _sun;
        private Light _fillLight;
        private ParticleSystem _ambientFx;
        private RaceConfigModel _config;
        private System.IDisposable _subscription;
        private float _flash;

        [Inject]
        public void Construct(RaceConfigModel config, ISubscriber<RaceStartedMessage> raceStarted)
        {
            _config = config;
            _subscription = raceStarted.Subscribe(OnRaceStarted);
        }

        private void Awake()
        {
            SetupCamera();
            SetupVolume();
            SetupEnvironment();
            SetupSun();
            SetupAmbientFx();
        }

        private void Start()
        {
            if (_config != null)
            {
                _config.Changed += OnConfigChanged;
                OnConfigChanged();
            }
        }

        private void Update()
        {
            if (_flash <= 0f)
            {
                return;
            }
            _flash = Mathf.Max(0f, _flash - Time.deltaTime * FLASH_DECAY_PER_SECOND);
            if (_colorAdjustments != null)
            {
                _colorAdjustments.postExposure.value = BASE_EXPOSURE + _flash;
            }
        }

        private void OnDestroy()
        {
            _subscription?.Dispose();
            if (_config != null)
            {
                _config.Changed -= OnConfigChanged;
            }
            if (_profile != null)
            {
                Destroy(_profile);
            }
            if (_skyboxMaterial != null)
            {
                Destroy(_skyboxMaterial);
            }
        }

        private void OnRaceStarted(RaceStartedMessage message)
        {
            _flash = START_FLASH;
        }

        private void OnConfigChanged()
        {
            ApplyMood(Systems_MapCatalog.Get(_config.SelectedMapIndex).Kind);
        }

        private void ApplyMood(TrackKind kind)
        {
            if (kind == TrackKind.Swamp)
            {
                SetSun(intensity: 1.05f, temperature: 6800f, rotation: Quaternion.Euler(55f, -20f, 0f));
                SetSky(new Color(0.42f, 0.52f, 0.45f), 1.4f);
                SetFog(new Color(0.45f, 0.52f, 0.4f), 25f, 90f);
                SetAmbient(new Color(0.42f, 0.5f, 0.42f), new Color(0.34f, 0.38f, 0.3f), new Color(0.16f, 0.18f, 0.12f));
                SetGrade(new Color(0.88f, 1f, 0.86f), saturation: -2f, vignetteIntensity: 0.32f);
                // Fireflies: slow-rising green motes that wander.
                SetAmbientFx(new Color(0.75f, 0.95f, 0.4f, 0.5f), rate: 26f,
                    sizeMin: 0.04f, sizeMax: 0.1f, riseSpeed: 0.25f, windX: 0f, wander: 0.6f);
            }
            else if (kind == TrackKind.Lumpy)
            {
                SetSun(intensity: 1.2f, temperature: 4300f, rotation: Quaternion.Euler(24f, -55f, 0f));
                SetSky(new Color(0.75f, 0.55f, 0.42f), 1.6f);
                SetFog(new Color(0.78f, 0.62f, 0.5f), 40f, 130f);
                SetAmbient(new Color(0.66f, 0.52f, 0.42f), new Color(0.5f, 0.42f, 0.38f), new Color(0.26f, 0.2f, 0.17f));
                SetGrade(new Color(1f, 0.94f, 0.85f), saturation: 14f, vignetteIntensity: 0.26f);
                // Wind-blown dust streaking across the valley.
                SetAmbientFx(new Color(0.85f, 0.7f, 0.5f, 0.28f), rate: 18f,
                    sizeMin: 0.15f, sizeMax: 0.5f, riseSpeed: 0.05f, windX: 3f, wander: 0.2f);
            }
            else
            {
                SetSun(intensity: 1.35f, temperature: 5600f, rotation: Quaternion.Euler(42f, -35f, 0f));
                SetSky(new Color(0.45f, 0.65f, 0.95f), 0.9f);
                SetFog(new Color(0.62f, 0.7f, 0.82f), 45f, 140f);
                SetAmbient(new Color(0.55f, 0.62f, 0.75f), new Color(0.42f, 0.42f, 0.45f), new Color(0.22f, 0.2f, 0.19f));
                SetGrade(Color.white, saturation: 10f, vignetteIntensity: 0.22f);
                // Sparse drifting pollen catches the sunlight.
                SetAmbientFx(new Color(1f, 1f, 0.9f, 0.3f), rate: 10f,
                    sizeMin: 0.03f, sizeMax: 0.08f, riseSpeed: -0.1f, windX: 0.6f, wander: 0.3f);
            }
        }

        private void SetSun(float intensity, float temperature, Quaternion rotation)
        {
            if (_sun == null)
            {
                return;
            }
            _sun.intensity = intensity;
            _sun.colorTemperature = temperature;
            _sun.transform.rotation = rotation;
            if (_fillLight != null)
            {
                // Fill mirrors the sun so shadow sides keep shape instead of
                // flattening into the ambient color.
                _fillLight.intensity = intensity * 0.25f;
                _fillLight.colorTemperature = temperature + 1500f;
                _fillLight.transform.rotation = Quaternion.Euler(
                    18f, rotation.eulerAngles.y + 180f, 0f);
            }
        }

        private void SetSky(Color tint, float atmosphere)
        {
            if (_skyboxMaterial == null)
            {
                return;
            }
            _skyboxMaterial.SetColor("_SkyTint", tint);
            _skyboxMaterial.SetFloat("_AtmosphereThickness", atmosphere);
        }

        private static void SetFog(Color color, float start, float end)
        {
            RenderSettings.fogColor = color;
            RenderSettings.fogStartDistance = start;
            RenderSettings.fogEndDistance = end;
        }

        private static void SetAmbient(Color sky, Color equator, Color ground)
        {
            RenderSettings.ambientSkyColor = sky;
            RenderSettings.ambientEquatorColor = equator;
            RenderSettings.ambientGroundColor = ground;
        }

        private void SetGrade(Color filter, float saturation, float vignetteIntensity)
        {
            if (_colorAdjustments != null)
            {
                _colorAdjustments.colorFilter.value = filter;
                _colorAdjustments.saturation.value = saturation;
            }
            if (_vignette != null)
            {
                _vignette.intensity.value = vignetteIntensity;
            }
        }

        private void SetAmbientFx(Color color, float rate, float sizeMin, float sizeMax,
            float riseSpeed, float windX, float wander)
        {
            if (_ambientFx == null)
            {
                return;
            }
            ParticleSystem.MainModule main = _ambientFx.main;
            main.startColor = color;
            main.startSize = new ParticleSystem.MinMaxCurve(sizeMin, sizeMax);
            ParticleSystem.EmissionModule emission = _ambientFx.emission;
            emission.rateOverTime = rate;
            ParticleSystem.VelocityOverLifetimeModule velocity = _ambientFx.velocityOverLifetime;
            velocity.enabled = true;
            // All three axes must use the same curve mode (TwoConstants here).
            velocity.x = new ParticleSystem.MinMaxCurve(windX * 0.5f, windX);
            velocity.y = new ParticleSystem.MinMaxCurve(riseSpeed * 0.5f, riseSpeed);
            velocity.z = new ParticleSystem.MinMaxCurve(0f, 0f);
            ParticleSystem.NoiseModule noise = _ambientFx.noise;
            noise.enabled = wander > 0f;
            noise.strength = wander;
            noise.frequency = 0.2f;
            _ambientFx.Clear();
        }

        private static void SetupCamera()
        {
            Camera mainCamera = Camera.main;
            if (mainCamera == null)
            {
                return;
            }
            UniversalAdditionalCameraData cameraData = mainCamera.GetUniversalAdditionalCameraData();
            cameraData.renderPostProcessing = true;
            cameraData.antialiasing = AntialiasingMode.SubpixelMorphologicalAntiAliasing;
            cameraData.antialiasingQuality = AntialiasingQuality.High;
        }

        private void SetupVolume()
        {
            _profile = ScriptableObject.CreateInstance<VolumeProfile>();

            Bloom bloom = _profile.Add<Bloom>(true);
            bloom.threshold.value = 0.85f;
            bloom.intensity.value = 0.75f;
            bloom.scatter.value = 0.65f;

            Tonemapping tonemapping = _profile.Add<Tonemapping>(true);
            tonemapping.mode.value = TonemappingMode.ACES;

            _vignette = _profile.Add<Vignette>(true);
            _vignette.intensity.value = 0.24f;
            _vignette.smoothness.value = 0.45f;

            ChromaticAberration aberration = _profile.Add<ChromaticAberration>(true);
            aberration.intensity.value = 0.08f;

            MotionBlur motionBlur = _profile.Add<MotionBlur>(true);
            motionBlur.intensity.value = 0.22f;

            _colorAdjustments = _profile.Add<ColorAdjustments>(true);
            _colorAdjustments.postExposure.value = BASE_EXPOSURE;
            _colorAdjustments.saturation.value = 12f;
            _colorAdjustments.contrast.value = 10f;
            _colorAdjustments.colorFilter.overrideState = true;

            var volume = gameObject.AddComponent<Volume>();
            volume.isGlobal = true;
            volume.priority = 10f;
            volume.profile = _profile;
        }

        private void SetupEnvironment()
        {
            Shader skyShader = Shader.Find("Skybox/Procedural");
            if (skyShader != null)
            {
                _skyboxMaterial = new Material(skyShader);
                _skyboxMaterial.SetFloat("_SunSize", 0.06f);
                _skyboxMaterial.SetFloat("_AtmosphereThickness", 0.9f);
                _skyboxMaterial.SetFloat("_Exposure", 1.25f);
                _skyboxMaterial.SetColor("_SkyTint", new Color(0.45f, 0.65f, 0.95f));
                _skyboxMaterial.SetColor("_GroundColor", new Color(0.28f, 0.27f, 0.26f));
                RenderSettings.skybox = _skyboxMaterial;
            }

            RenderSettings.ambientMode = AmbientMode.Trilight;
            SetAmbient(new Color(0.55f, 0.62f, 0.75f), new Color(0.42f, 0.42f, 0.45f), new Color(0.22f, 0.2f, 0.19f));
            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.Linear;
            SetFog(new Color(0.62f, 0.7f, 0.82f), 45f, 140f);
        }

        private void SetupSun()
        {
            Light sun = RenderSettings.sun;
            if (sun == null)
            {
                // Lighting settings has no explicit sun assigned in this scene;
                // one-time Awake lookup only, never repeated per frame.
                Light[] lights = FindObjectsByType<Light>(FindObjectsSortMode.None);
                for (int lightIndex = 0; lightIndex < lights.Length; lightIndex++)
                {
                    if (lights[lightIndex].type == LightType.Directional)
                    {
                        sun = lights[lightIndex];
                        break;
                    }
                }
            }
            if (sun == null)
            {
                return;
            }
            _sun = sun;
            sun.shadows = LightShadows.Soft;
            sun.shadowStrength = 0.9f;
            sun.shadowBias = 0.02f;
            sun.shadowNormalBias = 0.35f;
            sun.shadowNearPlane = 0.1f;
            sun.intensity = 1.4f;
            sun.useColorTemperature = true;
            sun.colorTemperature = 5600f;
            sun.transform.rotation = Quaternion.Euler(42f, -35f, 0f);

            var fillGo = new GameObject("FillLight");
            fillGo.transform.SetParent(transform, false);
            _fillLight = fillGo.AddComponent<Light>();
            _fillLight.type = LightType.Directional;
            _fillLight.shadows = LightShadows.None;
            _fillLight.useColorTemperature = true;
            _fillLight.intensity = 0.38f;
            _fillLight.colorTemperature = 7100f;
            _fillLight.transform.rotation = Quaternion.Euler(18f, 145f, 0f);

            var pipeline = GraphicsSettings.currentRenderPipeline as UniversalRenderPipelineAsset;
            if (pipeline != null)
            {
                pipeline.shadowDistance = SHADOW_DISTANCE;
                pipeline.shadowCascadeCount = 4;
            }
        }

        private void SetupAmbientFx()
        {
            Material particleMaterial = FxUtil.SoftParticleMaterial();
            if (particleMaterial == null)
            {
                return;
            }
            var go = new GameObject("AmbientFx");
            go.transform.SetParent(transform, false);
            // Hangs over the whole track area regardless of camera motion.
            go.transform.position = new Vector3(0f, 4f, 10f);
            var ps = go.AddComponent<ParticleSystem>();

            ParticleSystem.MainModule main = ps.main;
            main.simulationSpace = ParticleSystemSimulationSpace.World;
            main.startLifetime = new ParticleSystem.MinMaxCurve(6f, 12f);
            main.startSpeed = 0f;
            main.maxParticles = 400;

            ParticleSystem.ShapeModule shape = ps.shape;
            shape.shapeType = ParticleSystemShapeType.Box;
            shape.scale = new Vector3(50f, 8f, 46f);

            ParticleSystem.ColorOverLifetimeModule colorOverLifetime = ps.colorOverLifetime;
            colorOverLifetime.enabled = true;
            var gradient = new Gradient();
            gradient.SetKeys(
                new[]
                {
                    new GradientColorKey(Color.white, 0f),
                    new GradientColorKey(Color.white, 1f)
                },
                new[]
                {
                    new GradientAlphaKey(0f, 0f),
                    new GradientAlphaKey(1f, 0.2f),
                    new GradientAlphaKey(1f, 0.8f),
                    new GradientAlphaKey(0f, 1f)
                });
            colorOverLifetime.color = gradient;

            var ambientRenderer = ps.GetComponent<ParticleSystemRenderer>();
            ambientRenderer.material = particleMaterial;
            ambientRenderer.shadowCastingMode = ShadowCastingMode.Off;
            ambientRenderer.receiveShadows = false;
            _ambientFx = ps;
        }
    }
}
