#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace PoRacer.EditorTools
{
    /// <summary>
    /// Headless capture utility for running from the command line (-batchmode -executeMethod).
    /// Renders an off-screen camera to a RenderTexture and writes a PNG screenshot to disk.
    /// </summary>
    public static class Editor_HeadlessCapture
    {
        private const string DEFAULT_SCENE = "Assets/Scenes/SCN_RACE_FLAT.unity";

        [MenuItem("PoRacer/Capture Scene Screenshot (Headless Test)")]
        public static void CaptureSceneScreenshot()
        {
            string scenePath = DEFAULT_SCENE;
            string outputPath = "Captures/headless_screenshot.png";
            int width = 720;
            int height = 1544; // standard 9:16 portrait resolution for PoRacer

            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "--scene" && i + 1 < args.Length)
                {
                    scenePath = args[i + 1];
                }
                else if (args[i] == "--output" && i + 1 < args.Length)
                {
                    outputPath = args[i + 1];
                }
                else if (args[i] == "--width" && i + 1 < args.Length && int.TryParse(args[i + 1], out int w))
                {
                    width = w;
                }
                else if (args[i] == "--height" && i + 1 < args.Length && int.TryParse(args[i + 1], out int h))
                {
                    height = h;
                }
            }

            Debug.Log($"[HeadlessCapture] Opening scene: {scenePath}");
            EditorSceneManager.OpenScene(scenePath, OpenSceneMode.Single);

            Camera cam = Camera.main;
            if (cam == null)
            {
                cam = UnityEngine.Object.FindFirstObjectByType<Camera>();
            }

            bool createdTempCam = false;
            if (cam == null)
            {
                Debug.LogWarning("[HeadlessCapture] No camera found in scene. Creating a temporary capture camera.");
                GameObject camGo = new GameObject("TempCaptureCamera");
                cam = camGo.AddComponent<Camera>();
                cam.transform.position = new Vector3(0, 5, -15);
                cam.transform.LookAt(new Vector3(0, 1, 0));
                createdTempCam = true;
            }

            Debug.Log($"[HeadlessCapture] Capturing with camera '{cam.name}' at {width}x{height} to '{outputPath}'...");

            string dir = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
            }

            RenderTexture rt = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
            RenderTexture prevRt = RenderTexture.active;
            RenderTexture prevTarget = cam.targetTexture;

            try
            {
                cam.targetTexture = rt;
                cam.Render();

                RenderTexture.active = rt;
                Texture2D tex = new Texture2D(width, height, TextureFormat.RGB24, false);
                tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
                tex.Apply();

                byte[] pngBytes = tex.EncodeToPNG();
                File.WriteAllBytes(outputPath, pngBytes);
                Debug.Log($"[HeadlessCapture] SUCCESS: Screenshot written to {Path.GetFullPath(outputPath)} ({pngBytes.Length} bytes)");
                UnityEngine.Object.DestroyImmediate(tex);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[HeadlessCapture] ERROR capturing screenshot: {ex}");
                throw;
            }
            finally
            {
                cam.targetTexture = prevTarget;
                RenderTexture.active = prevRt;
                rt.Release();
                UnityEngine.Object.DestroyImmediate(rt);
                if (createdTempCam && cam != null)
                {
                    UnityEngine.Object.DestroyImmediate(cam.gameObject);
                }
            }
        }
    }
}
#endif
