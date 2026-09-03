using MessagePipe;
using PoRacer.Models;
using PoRacer.Presentation;
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
    ///
    /// Lighting depth comes from three code-generated pieces, none of which needs
    /// a bake or an asset:
    ///
    /// - A drifting cloud cookie on the sun, so the arena is not lit by a flat
    ///   infinite plane of light and the ground has moving contrast to read against.
    /// - Shadow distance that follows the shot. The pack camera needs 70 m of range
    ///   and gets soft, low-resolution cascades; a chase shot needs 20 m, and
    ///   spending the same shadow map over that range makes it several times
    ///   sharper for no extra cost.
    /// - A wetness global consumed by SH_TrackGrid, so the swamp ground darkens and
    ///   picks up a specular sheen without a second ground material.
    ///
    /// Adaptive Probe Volumes were considered and left off: the track is generated
    /// at runtime, so there is no static geometry to bake probes against, and
    /// switching the probe system over without a bake would lose the trilight
    /// ambient this relies on.
    /// </summary>
    public sealed class PostFxView : MonoBehaviour
    {
        private const float BASE_EXPOSURE = 0.15f;
        private const float START_FLASH = 0.85f;
        private const float FLASH_DECAY_PER_SECOND = 2.2f;
        private const float SHADOW_DISTANCE = 70f;
        // Range a tight chase shot needs. Below this the near cascade starts
        // clipping shadows off the racer the camera is actually on.
        private const float SHADOW_DISTANCE_CLOSE = 22f;
        // How much further than the subject the shadow range has to reach for the
        // scenery behind it to still cast.
        private const float SHADOW_RANGE_MULTIPLIER = 2.4f;
        private const float SHADOW_ADAPT_RATE = 25f;
        // Cascades are only worth their cost over a long range.
        private const float CASCADE_SPLIT_DISTANCE = 45f;
        private const float SHADOW_POLL_SECONDS = 0.2f;
        // Cloud cookie: metres across, and how fast the cover drifts.
        private const float COOKIE_SIZE_METERS = 220f;
        private const int COOKIE_RESOLUTION = 256;
        private const float COOKIE_DRIFT_METERS_PER_SECOND = 1.6f;

        // --- Shot-driven look ---------------------------------------------------
        // A wide pack shot wants everything legible; a chase shot is allowed to be
        // a photograph. These are the two ends the grade blends between.
        private const float WIDE_CONTRAST = 10f;
        private const float CHASE_CONTRAST = 22f;
        private const float WIDE_VIGNETTE = 0.22f;
        private const float CHASE_VIGNETTE = 0.34f;
        private const float SHOT_BLEND_RATE = 1.8f;
        // Gaussian depth of field: everything past the subject softens. Cheaper
        // than bokeh by a wide margin, and on a 9:16 handset screen the difference
        // is not visible.
        private const float DOF_MAX_RADIUS = 1.1f;
        private const float DOF_FOCUS_MARGIN = 4f;
        private const float DOF_FALLOFF_METERS = 18f;
        // Wide shots park the blur beyond anything on the track.
        private const float DOF_DISABLED_START = 400f;
        // Winner crossing: a brighter, shorter punch than the race-start flash,
        // with the aberration pushed to smear the edges of the frame.
        private const float WIN_FLASH = 1.25f;
        private const float WIN_ABERRATION = 0.55f;
        private const float BASE_ABERRATION = 0.08f;
        private const float ABERRATION_DECAY_PER_SECOND = 1.6f;

        private VolumeProfile _profile;
        private Material _skyboxMaterial;
        private Vignette _vignette;
        private ColorAdjustments _colorAdjustments;
        private Light _sun;
        private Light _fillLight;
        private ParticleSystem _ambientFx;
        private RaceConfigModel _config;
        private Systems_CameraDirector _cameraDirector;
        private System.IDisposable _subscription;
        private float _flash;
        private Texture2D _cloudCookie;
        private UniversalAdditionalLightData _sunData;
        private Camera _mainCamera;
        private float _cookieDrift;
        private float _shadowDistance = SHADOW_DISTANCE;
        private float _shadowPollTimer;
        private UniversalRenderPipelineAsset _pipeline;
        private DepthOfField _depthOfField;
        private ChromaticAberration _aberration;
        private System.IDisposable _finishSubscription;
        // 0 = wide pack shot, 1 = locked on a single racer. Everything cinematic
        // hangs off this one number so the look can never disagree with itself.
        private float _shotCloseness;
        private float _aberrationPulse;
        private float _mapContrast = WIDE_CONTRAST;
        private float _mapVignette = WIDE_VIGNETTE;

        private static readonly int WetnessId = Shader.PropertyToID("_PoRacerWetness");

        [Inject]
        public void Construct(RaceConfigModel config, Systems_CameraDirector cameraDirector,
            ISubscriber<RaceStartedMessage> raceStarted,
            ISubscriber<RacerFinishedMessage> racerFinished)
        {
            _config = config;
            _cameraDirector = cameraDirector;
            _subscription = raceStarted.Subscribe(OnRaceStarted);
            _finishSubscription = racerFinished.Subscribe(OnRacerFinished);
        }

        private void Awake()
        {
            _mainCamera = Camera.main;
            _pipeline = GraphicsSettings.currentRenderPipeline as UniversalRenderPipelineAsset;
            SetupCamera();
            SetupVolume();
            SetupEnvironment();
            SetupSun();
            SetupCloudCookie();
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
            float deltaTime = Time.deltaTime;
            DriftClouds(deltaTime);
            AdaptShadowRange(deltaTime);
            UpdateShotLook(deltaTime);

            if (_aberrationPulse > 0f)
            {
                _aberrationPulse = Mathf.Max(
                    0f, _aberrationPulse - deltaTime * ABERRATION_DECAY_PER_SECOND);
                if (_aberration != null)
                {
                    _aberration.intensity.value = BASE_ABERRATION
                        + (WIN_ABERRATION - BASE_ABERRATION) * _aberrationPulse;
                }
            }

            if (_flash <= 0f)
            {
                return;
            }
            _flash = Mathf.Max(0f, _flash - deltaTime * FLASH_DECAY_PER_SECOND);
            if (_colorAdjustments != null)
            {
                _colorAdjustments.postExposure.value = BASE_EXPOSURE + _flash;
            }
        }

        /// <summary>
        /// Blends the grade and the depth of field toward whichever shot the
        /// director is running.
        ///
        /// On a chase shot the frame is about one racer, so the background is
        /// allowed to fall away and the contrast and vignette can close in. On the
        /// wide pack shot every racer is the subject, so the blur parks beyond the
        /// track and the grade opens back up. Blending rather than switching keeps
        /// a cut from reading as a glitch.
        ///
        /// The per-map mood set by ApplyMood stays the baseline; this only ever
        /// pushes away from it and returns.
        /// </summary>
        private void UpdateShotLook(float deltaTime)
        {
            Transform target = _cameraDirector != null ? _cameraDirector.ActiveShotTarget : null;
            bool close = target != null && _mainCamera != null;
            _shotCloseness = Mathf.MoveTowards(
                _shotCloseness, close ? 1f : 0f, deltaTime * SHOT_BLEND_RATE);

            if (_colorAdjustments != null)
            {
                _colorAdjustments.contrast.value =
                    Mathf.Lerp(_mapContrast, CHASE_CONTRAST, _shotCloseness);
            }
            if (_vignette != null)
            {
                _vignette.intensity.value =
                    Mathf.Lerp(_mapVignette, CHASE_VIGNETTE, _shotCloseness);
            }
            if (_depthOfField == null)
            {
                return;
            }

            // Focus sits a little past the subject so the racer itself is never
            // caught in the near edge of the falloff.
            float focusStart = DOF_DISABLED_START;
            if (close)
            {
                float subjectDistance = Vector3.Distance(
                    _mainCamera.transform.position, target.position);
                focusStart = subjectDistance + DOF_FOCUS_MARGIN;
            }
            // Blend the start outward as the shot opens up, so pulling back to the
            // pack camera dissolves the blur instead of popping it off.
            float blendedStart = Mathf.Lerp(DOF_DISABLED_START, focusStart, _shotCloseness);
            _depthOfField.gaussianStart.value = blendedStart;
            _depthOfField.gaussianEnd.value = blendedStart + DOF_FALLOFF_METERS;
            _depthOfField.gaussianMaxRadius.value = DOF_MAX_RADIUS * _shotCloseness;
        }

        /// <summary>
        /// The winner crossing gets its own punch: brighter than the start flash and
        /// with the aberration pushed hard for a beat, so the moment reads even in a
        /// clip with no audio.
        /// </summary>
        private void OnRacerFinished(RacerFinishedMessage message)
        {
            if (message.Place != 1)
            {
                return;
            }
            _flash = WIN_FLASH;
            _aberrationPulse = 1f;
        }

        /// <summary>
        /// Slides the cloud cookie across the arena. Offsetting the cookie rather
        /// than rotating the sun keeps the key light direction — and therefore the
        /// whole grade — exactly where the map set it.
        /// </summary>
        private void DriftClouds(float deltaTime)
        {
            if (_sunData == null)
            {
                return;
            }
            _cookieDrift += deltaTime * COOKIE_DRIFT_METERS_PER_SECOND;
            if (_cookieDrift > COOKIE_SIZE_METERS)
            {
                // Wrap on the cookie size so the offset never grows without bound
                // and starts losing float precision during a long session.
                _cookieDrift -= COOKIE_SIZE_METERS;
            }
            _sunData.lightCookieOffset = new Vector2(_cookieDrift, _cookieDrift * 0.35f);
        }

        /// <summary>
        /// Matches the shadow range to the shot. A chase camera locked onto one
        /// racer only needs shadows out to a couple of body lengths past it, and
        /// concentrating the same shadow map over that range is what makes the
        /// contact shadows under a creature look sharp instead of mushy.
        /// </summary>
        private void AdaptShadowRange(float deltaTime)
        {
            if (_pipeline == null)
            {
                return;
            }
            _shadowPollTimer -= deltaTime;
            if (_shadowPollTimer <= 0f)
            {
                _shadowPollTimer = SHADOW_POLL_SECONDS;
                _shadowDistance = Mathf.MoveTowards(
                    _shadowDistance, TargetShadowDistance(), SHADOW_ADAPT_RATE * SHADOW_POLL_SECONDS);
                _pipeline.shadowDistance = _shadowDistance;
                // Four cascades over 20 m wastes three of them on empty range.
                _pipeline.shadowCascadeCount = _shadowDistance > CASCADE_SPLIT_DISTANCE ? 4 : 2;
            }
        }

        private float TargetShadowDistance()
        {
            Transform target = _cameraDirector != null ? _cameraDirector.ActiveShotTarget : null;
            if (target == null || _mainCamera == null)
            {
                // Wide shot: the whole pack has to cast, so take the full range.
                return SHADOW_DISTANCE;
            }
            float subjectDistance = Vector3.Distance(_mainCamera.transform.position, target.position);
            return Mathf.Clamp(
                subjectDistance * SHADOW_RANGE_MULTIPLIER, SHADOW_DISTANCE_CLOSE, SHADOW_DISTANCE);
        }

        private void OnDestroy()
        {
            _subscription?.Dispose();
            _finishSubscription?.Dispose();
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
            if (_cloudCookie != null)
            {
                Destroy(_cloudCookie);
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
            // Read by SH_TrackGrid. A global rather than a material property: the
            // track builder spawns its ground chunks at race time, so there is no
            // single material instance to write to up front.
            Shader.SetGlobalFloat(WetnessId, kind == TrackKind.Swamp ? 0.85f : 0f);

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
            // Stored rather than written straight through: UpdateShotLook blends
            // away from this baseline every frame, so writing the vignette here
            // would be overwritten before it was ever seen.
            _mapVignette = vignetteIntensity;
            _mapContrast = WIDE_CONTRAST;
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

            _aberration = _profile.Add<ChromaticAberration>(true);
            _aberration.intensity.value = BASE_ABERRATION;

            // Gaussian rather than bokeh: a fraction of the cost, and the shape of
            // the out-of-focus highlights is not what sells a 9:16 phone frame.
            _depthOfField = _profile.Add<DepthOfField>(true);
            _depthOfField.mode.value = DepthOfFieldMode.Gaussian;
            _depthOfField.gaussianStart.value = DOF_DISABLED_START;
            _depthOfField.gaussianEnd.value = DOF_DISABLED_START + DOF_FALLOFF_METERS;
            // Starts at zero radius: the opening shot is wide, so no blur at all.
            _depthOfField.gaussianMaxRadius.value = 0f;

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
                Light[] lights = FindObjectsByType<Light>();
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

            if (_pipeline != null)
            {
                // Starting point; AdaptShadowRange takes it from here per shot.
                _shadowDistance = SHADOW_DISTANCE;
                _pipeline.shadowDistance = SHADOW_DISTANCE;
                _pipeline.shadowCascadeCount = 4;
            }
        }

        /// <summary>
        /// Builds the drifting cloud cover as a light cookie.
        ///
        /// Without it the arena is lit by a mathematically perfect infinite plane
        /// of light, which is what makes an open procedural track read as flat: the
        /// ground has no large-scale luminance variation for the eye to use. The
        /// cookie is a few octaves of value noise, biased bright so the default
        /// state is open sun with occasional shade rather than permanent overcast.
        ///
        /// Generated once at startup into a single-channel texture and never
        /// touched again — only its offset animates.
        /// </summary>
        private void SetupCloudCookie()
        {
            if (_sun == null || SystemInfo.graphicsDeviceType == GraphicsDeviceType.Null)
            {
                return;
            }

            _cloudCookie = new Texture2D(
                COOKIE_RESOLUTION, COOKIE_RESOLUTION, TextureFormat.R8, mipChain: true)
            {
                name = "CloudCookie",
                // The cookie tiles as the offset walks past its edge, so both axes
                // have to repeat or a visible seam sweeps across the track.
                wrapMode = TextureWrapMode.Repeat,
                filterMode = FilterMode.Bilinear
            };

            var pixels = new Color32[COOKIE_RESOLUTION * COOKIE_RESOLUTION];
            for (int y = 0; y < COOKIE_RESOLUTION; y++)
            {
                for (int x = 0; x < COOKIE_RESOLUTION; x++)
                {
                    // Perlin sampled on a whole number of periods so the pattern
                    // wraps cleanly at the texture edge.
                    float u = (float)x / COOKIE_RESOLUTION;
                    float v = (float)y / COOKIE_RESOLUTION;
                    float cover =
                        TileableNoise(u, v, 3f) * 0.55f +
                        TileableNoise(u, v, 7f) * 0.30f +
                        TileableNoise(u, v, 17f) * 0.15f;
                    // Bias toward open sky: clouds should punctuate the light, not
                    // define it, or the whole race sits in shade.
                    float light = Mathf.Clamp01(0.55f + cover * 0.85f);
                    byte level = (byte)(light * 255f);
                    pixels[y * COOKIE_RESOLUTION + x] = new Color32(level, level, level, 255);
                }
            }
            _cloudCookie.SetPixels32(pixels);
            _cloudCookie.Apply();

            _sun.cookie = _cloudCookie;
            _sunData = _sun.GetUniversalAdditionalLightData();
            if (_sunData != null)
            {
                _sunData.lightCookieSize = new Vector2(COOKIE_SIZE_METERS, COOKIE_SIZE_METERS);
                _sunData.lightCookieOffset = Vector2.zero;
            }
        }

        /// <summary>
        /// Value noise that tiles over the unit square, by sampling Perlin on a
        /// circle in each axis. A straight Perlin lookup would seam at the wrap.
        /// </summary>
        private static float TileableNoise(float u, float v, float frequency)
        {
            float angleU = u * Mathf.PI * 2f;
            float angleV = v * Mathf.PI * 2f;
            float x = (Mathf.Cos(angleU) + 1f) * frequency;
            float y = (Mathf.Sin(angleU) + 1f) * frequency;
            float z = (Mathf.Cos(angleV) + 1f) * frequency;
            float w = (Mathf.Sin(angleV) + 1f) * frequency;
            // Two 2D lookups averaged: cheap, and enough structure for cloud cover.
            return (Mathf.PerlinNoise(x, z) + Mathf.PerlinNoise(y, w)) * 0.5f - 0.5f;
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
