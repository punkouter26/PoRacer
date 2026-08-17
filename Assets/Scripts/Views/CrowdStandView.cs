using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Excitement level shared by every grandstand: views with race messages
    /// call Excite, stands read Level. Time-based so there is no decay state to
    /// tick — Level fades to zero over a few seconds after the last event.
    /// </summary>
    internal static class CrowdMood
    {
        private const float DECAY_SECONDS = 3f;

        private static float _lastExciteTime = -100f;

        public static void Excite() => _lastExciteTime = Time.time;

        public static float Level => Mathf.Clamp01(1f - (Time.time - _lastExciteTime) / DECAY_SECONDS);
    }

    /// <summary>
    /// Animates one grandstand's crowd block built by Systems_TrackBuilder: a
    /// gentle idle sway, and a hard bounce while CrowdMood is hot (race start,
    /// lead changes, wins). The crowd is a single vertex-colored mesh, so each
    /// stand is one draw call and the animation is one transform write.
    /// </summary>
    public sealed class CrowdStandView : MonoBehaviour
    {
        private const float IDLE_AMPLITUDE = 0.02f;
        private const float IDLE_SPEED = 1.6f;
        private const float EXCITED_AMPLITUDE = 0.12f;
        private const float EXCITED_SPEED = 9f;

        private Transform _crowd;
        private float _phase;

        /// <summary>Track builder hands over the crowd block to animate.</summary>
        public void Initialize(Transform crowd, float phaseOffset)
        {
            _crowd = crowd;
            _phase = phaseOffset;
        }

        private void Update()
        {
            if (_crowd == null)
            {
                return;
            }
            float excitement = CrowdMood.Level;
            float amplitude = Mathf.Lerp(IDLE_AMPLITUDE, EXCITED_AMPLITUDE, excitement);
            float speed = Mathf.Lerp(IDLE_SPEED, EXCITED_SPEED, excitement);
            // Abs(sin) reads as jumping: sharp at the bottom, floaty at the top.
            float bounce = Mathf.Abs(Mathf.Sin((Time.time + _phase) * speed));
            _crowd.localScale = new Vector3(1f, 1f + amplitude * bounce, 1f);
        }

        /// <summary>
        /// One crowd block: rows of small colored quads as a single mesh with
        /// vertex colors. Faces local +z; the parent stand orients it at the track.
        /// </summary>
        public static GameObject BuildCrowdMesh(System.Random rng, int rows, int columns)
        {
            var crowd = new GameObject("Crowd");
            int quadCount = rows * columns;
            var vertices = new Vector3[quadCount * 4];
            var colors = new Color[quadCount * 4];
            var triangles = new int[quadCount * 6];
            const float personWidth = 0.34f;
            const float personHeight = 0.5f;
            const float seatWidth = 0.55f;
            const float rowDepth = 0.8f;
            const float rowRise = 0.45f;

            int quadIndex = 0;
            for (int rowIndex = 0; rowIndex < rows; rowIndex++)
            {
                for (int columnIndex = 0; columnIndex < columns; columnIndex++)
                {
                    float x = (columnIndex - (columns - 1) * 0.5f) * seatWidth
                        + ((float)rng.NextDouble() - 0.5f) * 0.12f;
                    float y = rowIndex * rowRise + (float)rng.NextDouble() * 0.06f;
                    float z = -rowIndex * rowDepth;
                    int vertexBase = quadIndex * 4;
                    vertices[vertexBase + 0] = new Vector3(x - personWidth * 0.5f, y, z);
                    vertices[vertexBase + 1] = new Vector3(x + personWidth * 0.5f, y, z);
                    vertices[vertexBase + 2] = new Vector3(x - personWidth * 0.5f, y + personHeight, z);
                    vertices[vertexBase + 3] = new Vector3(x + personWidth * 0.5f, y + personHeight, z);
                    Color color = Color.HSVToRGB((float)rng.NextDouble(), 0.65f, 0.95f);
                    colors[vertexBase + 0] = color;
                    colors[vertexBase + 1] = color;
                    colors[vertexBase + 2] = color;
                    colors[vertexBase + 3] = color;
                    int triangleBase = quadIndex * 6;
                    triangles[triangleBase + 0] = vertexBase + 0;
                    triangles[triangleBase + 1] = vertexBase + 2;
                    triangles[triangleBase + 2] = vertexBase + 1;
                    triangles[triangleBase + 3] = vertexBase + 1;
                    triangles[triangleBase + 4] = vertexBase + 2;
                    triangles[triangleBase + 5] = vertexBase + 3;
                    quadIndex++;
                }
            }

            var mesh = new Mesh { vertices = vertices, colors = colors, triangles = triangles };
            mesh.RecalculateBounds();
            crowd.AddComponent<MeshFilter>().sharedMesh = mesh;
            var crowdRenderer = crowd.AddComponent<MeshRenderer>();
            Shader spriteShader = Shader.Find("Sprites/Default");
            if (spriteShader != null)
            {
                crowdRenderer.sharedMaterial = GetCrowdMaterial(spriteShader);
            }
            crowdRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            crowdRenderer.receiveShadows = false;
            return crowd;
        }

        private static Material _crowdMaterial;

        private static Material GetCrowdMaterial(Shader spriteShader)
        {
            if (_crowdMaterial == null)
            {
                // Sprites/Default multiplies vertex color: one material, all hues.
                _crowdMaterial = new Material(spriteShader);
            }
            return _crowdMaterial;
        }
    }
}
