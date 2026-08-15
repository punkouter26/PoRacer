using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace PoRacer.Views
{
    /// <summary>
    /// Builds the scene look in code at startup: post-processing volume (bloom,
    /// ACES tonemapping, vignette, color grade), procedural skybox, tri-light
    /// ambient, and distance fog. No profile or material asset files needed.
    /// </summary>
    public sealed class PostFxView : MonoBehaviour
    {
        private VolumeProfile _profile;
        private Material _skyboxMaterial;

        private void Awake()
        {
            SetupCamera();
            SetupVolume();
            SetupEnvironment();
            SetupSun();
        }

        private void OnDestroy()
        {
            if (_profile != null)
            {
                Destroy(_profile);
            }
            if (_skyboxMaterial != null)
            {
                Destroy(_skyboxMaterial);
            }
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
            cameraData.antialiasing = AntialiasingMode.FastApproximateAntialiasing;
        }

        private void SetupVolume()
        {
            _profile = ScriptableObject.CreateInstance<VolumeProfile>();

            Bloom bloom = _profile.Add<Bloom>(true);
            bloom.threshold.value = 0.9f;
            bloom.intensity.value = 0.55f;
            bloom.scatter.value = 0.6f;

            Tonemapping tonemapping = _profile.Add<Tonemapping>(true);
            tonemapping.mode.value = TonemappingMode.ACES;

            Vignette vignette = _profile.Add<Vignette>(true);
            vignette.intensity.value = 0.22f;
            vignette.smoothness.value = 0.4f;

            ColorAdjustments colorAdjustments = _profile.Add<ColorAdjustments>(true);
            colorAdjustments.postExposure.value = 0.15f;
            colorAdjustments.saturation.value = 10f;
            colorAdjustments.contrast.value = 8f;

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
            RenderSettings.ambientSkyColor = new Color(0.55f, 0.62f, 0.75f);
            RenderSettings.ambientEquatorColor = new Color(0.42f, 0.42f, 0.45f);
            RenderSettings.ambientGroundColor = new Color(0.22f, 0.2f, 0.19f);

            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.Linear;
            RenderSettings.fogStartDistance = 45f;
            RenderSettings.fogEndDistance = 140f;
            RenderSettings.fogColor = new Color(0.62f, 0.7f, 0.82f);
        }

        private static void SetupSun()
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
            sun.shadows = LightShadows.Soft;
            sun.shadowStrength = 0.85f;
            sun.intensity = 1.35f;
            sun.useColorTemperature = true;
            sun.colorTemperature = 5600f;
            sun.transform.rotation = Quaternion.Euler(42f, -35f, 0f);
        }
    }
}
