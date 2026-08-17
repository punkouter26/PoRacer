using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Googly eyes: one shared eye-pair sprite on a small quad stuck to the
    /// front of each creature's root body, with a periodic squash blink.
    /// Texture and material are generated once and shared by every racer, so
    /// the whole grid of faces batches together. Added by Systems_Spawn after
    /// tinting so the eyes keep their own colors.
    /// </summary>
    public sealed class EyesView : MonoBehaviour
    {
        private const float BLINK_SECONDS = 0.12f;
        private const float BLINK_MIN_INTERVAL = 2f;
        private const float BLINK_MAX_INTERVAL = 6f;
        private const float FACE_SCALE = 0.42f;

        private static Texture2D _eyesTexture;
        private static Material _eyesMaterial;

        private Transform _face;
        private float _nextBlinkTime;
        private float _blinkUntil;
        private Vector3 _openScale;

        private void Start()
        {
            ArticulationBody[] bodies = GetComponentsInChildren<ArticulationBody>();
            ArticulationBody root = null;
            for (int bodyIndex = 0; bodyIndex < bodies.Length; bodyIndex++)
            {
                if (bodies[bodyIndex].isRoot)
                {
                    root = bodies[bodyIndex];
                    break;
                }
            }
            if (root == null)
            {
                enabled = false;
                return;
            }
            Renderer headRenderer = root.GetComponentInChildren<Renderer>();
            if (headRenderer == null)
            {
                enabled = false;
                return;
            }
            Material material = GetEyesMaterial();
            if (material == null)
            {
                enabled = false;
                return;
            }

            var face = GameObject.CreatePrimitive(PrimitiveType.Quad);
            face.name = "Eyes";
            Destroy(face.GetComponent<Collider>());
            // Anchor on the head's rendered bounds in WORLD space, then parent to
            // the root body so the face rides the physics. Local-offset math broke
            // on scaled or pre-rotated roots (snake, centipede): the eyes drifted
            // off the solid body. Racers stand upright at Start (spawn frame), so
            // world +z is the face direction; the quad's visible face has a -z
            // normal, hence the 180 turn.
            Bounds headBounds = headRenderer.bounds;
            face.transform.position = headBounds.center
                + new Vector3(0f, headBounds.extents.y * 0.45f, headBounds.extents.z + 0.012f);
            face.transform.rotation = Quaternion.Euler(0f, 180f, 0f);
            face.transform.localScale = new Vector3(FACE_SCALE, FACE_SCALE * 0.5f, 1f);
            face.transform.SetParent(root.transform, worldPositionStays: true);
            // Captured after parenting: the local scale now includes whatever
            // compensation the scaled root required.
            _openScale = face.transform.localScale;
            var faceRenderer = face.GetComponent<MeshRenderer>();
            faceRenderer.sharedMaterial = material;
            faceRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            faceRenderer.receiveShadows = false;
            _face = face.transform;
            _nextBlinkTime = Time.time + Random.Range(BLINK_MIN_INTERVAL, BLINK_MAX_INTERVAL);
        }

        private void Update()
        {
            if (_face == null)
            {
                return;
            }
            if (Time.time >= _nextBlinkTime)
            {
                _blinkUntil = Time.time + BLINK_SECONDS;
                _nextBlinkTime = Time.time + Random.Range(BLINK_MIN_INTERVAL, BLINK_MAX_INTERVAL);
            }
            bool blinking = Time.time < _blinkUntil;
            _face.localScale = blinking
                ? new Vector3(_openScale.x, _openScale.y * 0.12f, _openScale.z)
                : _openScale;
        }

        private static Material GetEyesMaterial()
        {
            if (_eyesMaterial != null)
            {
                return _eyesMaterial;
            }
            Shader spriteShader = Shader.Find("Sprites/Default");
            if (spriteShader == null)
            {
                return null;
            }
            _eyesMaterial = new Material(spriteShader)
            {
                mainTexture = GetEyesTexture()
            };
            return _eyesMaterial;
        }

        /// <summary>Two white googly eyes with off-center pupils, alpha elsewhere.</summary>
        private static Texture2D GetEyesTexture()
        {
            if (_eyesTexture != null)
            {
                return _eyesTexture;
            }
            const int width = 128;
            const int height = 64;
            var texture = new Texture2D(width, height, TextureFormat.RGBA32, false);
            var clear = new Color(0f, 0f, 0f, 0f);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    texture.SetPixel(x, y, clear);
                }
            }
            DrawEye(texture, centerX: 36, centerY: 32);
            DrawEye(texture, centerX: 92, centerY: 32);
            texture.Apply();
            _eyesTexture = texture;
            return texture;
        }

        private static void DrawEye(Texture2D texture, int centerX, int centerY)
        {
            const int whiteRadius = 24;
            const int pupilRadius = 10;
            // Pupils sit slightly up-and-inward: alert, a little unhinged.
            int pupilX = centerX + (centerX < 64 ? 5 : -5);
            int pupilY = centerY + 6;
            for (int y = centerY - whiteRadius; y <= centerY + whiteRadius; y++)
            {
                for (int x = centerX - whiteRadius; x <= centerX + whiteRadius; x++)
                {
                    float distance = Mathf.Sqrt((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY));
                    if (distance > whiteRadius)
                    {
                        continue;
                    }
                    float pupilDistance = Mathf.Sqrt((x - pupilX) * (x - pupilX) + (y - pupilY) * (y - pupilY));
                    Color color = pupilDistance <= pupilRadius ? Color.black : Color.white;
                    // Soft edge on the white so the eye doesn't alias hard.
                    color.a = Mathf.Clamp01(whiteRadius - distance);
                    texture.SetPixel(x, y, color);
                }
            }
        }
    }
}
