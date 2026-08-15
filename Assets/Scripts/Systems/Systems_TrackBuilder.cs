using System;
using UnityEngine;

namespace PoRacer.Systems
{
    public enum TrackKind
    {
        Flat = 0,
        Hills = 1,
        Bumps = 2,
        Walls = 3,
        // Training-only until brains are retrained on it.
        Rough = 4,
        // Rough terrain plus obstacle boxes: the shared training scene's hard mode.
        RoughBlocked = 5,
        // Player-selectable map 2: rough hills with chunky scattered objects to
        // steer around on the way to the finish line.
        Lumpy = 6,
        // Player-selectable map 3: flat ground with mud pits that drag racers
        // down and gate walls that force a path choice.
        Swamp = 7
    }

    /// <summary>
    /// Procedural track construction shared by the race scene and training areas.
    /// Builds ground (flat plane or a sine-hill mesh) plus scattered obstacles
    /// under a single parent that the caller owns and can clear/rebuild.
    /// </summary>
    public sealed class Systems_TrackBuilder
    {
        private const float HILL_AMPLITUDE = 0.35f;
        private const float HILL_WAVELENGTH = 6f;
        private const int MESH_STEPS_PER_METER = 2;
        private const float ROUGH_AMPLITUDE = 0.8f;
        private const float ROUGH_NOISE_SCALE = 0.35f;    // ~3 m terrain features
        private const float ROUGH_DETAIL_SCALE = 0.95f;   // ~1 m secondary bumps
        private const float ROUGH_SPAWN_PAD_END_Z = 6f;   // full roughness from here on
        private const float ROUGH_SPAWN_PAD_FLAT_Z = 2f;  // flat until here (racers spawn stable)
        private const float MARKER_ALPHA = 0.12f;
        private const float MARKER_HEIGHT_OFFSET = 0.02f;
        // Visual-only side aprons that hold trees, tents, and bushes.
        private const float DECOR_MARGIN = 14f;
        private const float ARCH_MODEL_WIDTH = 14f;

        private static Material _markerMaterial;
        private static Material _mudMaterial;
        private static readonly System.Collections.Generic.Dictionary<TrackKind, Material> GroundMaterials = new();
        private static bool _propsLoaded;
        private static GameObject _bushPrefab;
        private static GameObject _forestPrefab;
        private static GameObject _tentsPrefab;
        private static GameObject _archPrefab;
        // The imported .glb props reference a colormap texture that does not ship
        // with them, so they render white; these replacements are built in code.
        private static Material _propAtlasMaterial;
        private static Material _archMaterial;

        /// <summary>
        /// Curriculum knob (0..1+): scales rough-terrain height. Training sets this
        /// from the mlagents environment_parameters lesson; gameplay leaves it at 1.
        /// </summary>
        public static float RoughAmplitudeScale = 1f;

        private readonly Material _groundMaterial;
        private readonly Material _obstacleMaterial;
        private readonly PhysicsMaterial _physicsMaterial;

        public Systems_TrackBuilder(Material groundMaterial, Material obstacleMaterial, PhysicsMaterial physicsMaterial)
        {
            _groundMaterial = groundMaterial;
            _obstacleMaterial = obstacleMaterial;
            _physicsMaterial = physicsMaterial;
        }

        /// <summary>
        /// Clears previous children and builds the track under 'parent'. Origin is
        /// the start line center. 'decorate' adds visual-only side scenery and the
        /// finish arch (gameplay scene only; training keeps the bare track).
        /// </summary>
        public void Build(TrackKind kind, Transform parent, float width, float length, System.Random rng,
            bool decorate = false, float finishZ = -1f)
        {
            for (int childIndex = parent.childCount - 1; childIndex >= 0; childIndex--)
            {
                UnityEngine.Object.Destroy(parent.GetChild(childIndex).gameObject);
            }

            float sideMargin = decorate ? DECOR_MARGIN : 0f;
            if (kind == TrackKind.Hills || kind == TrackKind.Rough || kind == TrackKind.RoughBlocked || kind == TrackKind.Lumpy)
            {
                BuildTerrainMesh(kind, parent, width, length, sideMargin);
            }
            else
            {
                BuildFlatGround(kind, parent, width, length, sideMargin);
            }

            BuildScaleMarkers(kind, parent, width, length);

            if (kind == TrackKind.Bumps)
            {
                ScatterBoxes(kind, parent, width, length, rng, count: 24,
                    minSize: new Vector3(0.4f, 0.08f, 0.4f), maxSize: new Vector3(1.2f, 0.22f, 1.2f));
            }
            else if (kind == TrackKind.Walls || kind == TrackKind.RoughBlocked)
            {
                ScatterBoxes(kind, parent, width, length, rng, count: 10,
                    minSize: new Vector3(2f, 0.25f, 0.25f), maxSize: new Vector3(5f, 0.4f, 0.35f));
            }
            else if (kind == TrackKind.Lumpy)
            {
                ScatterBoxes(kind, parent, width, length, rng, count: 14,
                    minSize: new Vector3(0.5f, 0.3f, 0.5f), maxSize: new Vector3(1.8f, 0.9f, 1.8f));
            }
            else if (kind == TrackKind.Swamp)
            {
                BuildMudPits(parent, width, length, rng, count: 5);
                BuildGates(parent, width, length, rng, count: 2);
            }

            if (decorate)
            {
                DecorateTrack(kind, parent, width, length, finishZ > 0f ? finishZ : length - 2f, rng);
            }
        }

        /// <summary>Surface height at local z on the track centerline (x = 0).</summary>
        public static float SurfaceHeight(TrackKind kind, float z) => SurfaceHeight(kind, 0f, z);

        /// <summary>Surface height at a local (x, z) for the given kind — used to place goals/spawns.</summary>
        public static float SurfaceHeight(TrackKind kind, float x, float z)
        {
            if (kind == TrackKind.Hills)
            {
                return (Mathf.Sin(z / HILL_WAVELENGTH * 2f * Mathf.PI) * 0.5f + 0.5f) * HILL_AMPLITUDE;
            }
            if (kind == TrackKind.Rough || kind == TrackKind.RoughBlocked || kind == TrackKind.Lumpy)
            {
                // Deterministic Perlin terrain (same coords -> same height, per project
                // determinism rules). Two octaves: broad hills plus 1 m detail bumps.
                float broad = Mathf.PerlinNoise(x * ROUGH_NOISE_SCALE + 11.3f, z * ROUGH_NOISE_SCALE + 7.9f);
                float detail = Mathf.PerlinNoise(x * ROUGH_DETAIL_SCALE + 3.1f, z * ROUGH_DETAIL_SCALE + 17.7f);
                float signedHeight = (broad * 0.75f + detail * 0.25f - 0.5f) * 2f * ROUGH_AMPLITUDE;
                // Flat spawn pad: zero roughness behind the grid and through the first
                // metres, ramping to full so racers are born on stable ground.
                float ramp = Mathf.InverseLerp(ROUGH_SPAWN_PAD_FLAT_Z, ROUGH_SPAWN_PAD_END_Z, z);
                return signedHeight * ramp * RoughAmplitudeScale;
            }
            return 0f;
        }

        private void BuildFlatGround(TrackKind kind, Transform parent, float width, float length, float sideMargin)
        {
            float fullWidth = width + sideMargin * 2f;
            var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "Ground";
            ground.transform.SetParent(parent, false);
            // A plane primitive is 10x10 at scale 1; margin behind the start line.
            ground.transform.localPosition = new Vector3(0f, 0f, length * 0.5f - 3f);
            ground.transform.localScale = new Vector3(fullWidth / 10f, 1f, (length + 8f) / 10f);
            ground.GetComponent<MeshRenderer>().sharedMaterial = GroundMaterialFor(kind);
            ground.GetComponent<MeshCollider>().sharedMaterial = _physicsMaterial;
        }

        private void BuildTerrainMesh(TrackKind kind, Transform parent, float width, float length, float sideMargin)
        {
            width += sideMargin * 2f;
            float fullLength = length + 8f;
            float zStart = -5f;
            int stepsZ = Mathf.CeilToInt(fullLength * MESH_STEPS_PER_METER);
            // Hills only vary along z, so a coarse x grid suffices; Rough varies in
            // both axes with ~1 m features and needs the full resolution.
            int stepsX = kind == TrackKind.Rough || kind == TrackKind.RoughBlocked || kind == TrackKind.Lumpy
                ? Mathf.CeilToInt(width * MESH_STEPS_PER_METER)
                : Mathf.CeilToInt(width * MESH_STEPS_PER_METER / 4f);
            var vertices = new Vector3[(stepsX + 1) * (stepsZ + 1)];
            var uvs = new Vector2[vertices.Length];
            var triangles = new int[stepsX * stepsZ * 6];

            for (int zIndex = 0; zIndex <= stepsZ; zIndex++)
            {
                float z = zStart + fullLength * zIndex / stepsZ;
                for (int xIndex = 0; xIndex <= stepsX; xIndex++)
                {
                    float x = -width * 0.5f + width * xIndex / stepsX;
                    vertices[zIndex * (stepsX + 1) + xIndex] = new Vector3(x, SurfaceHeight(kind, x, z), z);
                    // One texture tile per 4 m, same density as the flat plane.
                    uvs[zIndex * (stepsX + 1) + xIndex] = new Vector2(x * 0.25f, z * 0.25f);
                }
            }
            int triangleIndex = 0;
            for (int zIndex = 0; zIndex < stepsZ; zIndex++)
            {
                for (int xIndex = 0; xIndex < stepsX; xIndex++)
                {
                    int corner = zIndex * (stepsX + 1) + xIndex;
                    triangles[triangleIndex++] = corner;
                    triangles[triangleIndex++] = corner + stepsX + 1;
                    triangles[triangleIndex++] = corner + 1;
                    triangles[triangleIndex++] = corner + 1;
                    triangles[triangleIndex++] = corner + stepsX + 1;
                    triangles[triangleIndex++] = corner + stepsX + 2;
                }
            }
            var mesh = new Mesh { vertices = vertices, uv = uvs, triangles = triangles };
            mesh.RecalculateNormals();

            var ground = new GameObject("Ground");
            ground.transform.SetParent(parent, false);
            ground.AddComponent<MeshFilter>().sharedMesh = mesh;
            ground.AddComponent<MeshRenderer>().sharedMaterial = GroundMaterialFor(kind);
            var collider = ground.AddComponent<MeshCollider>();
            collider.sharedMesh = mesh;
            collider.sharedMaterial = _physicsMaterial;
        }

        /// <summary>
        /// Checkerboard of 1 m x 1 m semi-transparent tiles so creature scale is
        /// readable at a glance. All tiles share one mesh and one material:
        /// a single draw call, no colliders, no shadows.
        /// </summary>
        private static void BuildScaleMarkers(TrackKind kind, Transform parent, float width, float length)
        {
            // Headless training builds strip rendering: no device, and Shader.Find
            // may return null for shaders not in Always Included Shaders. Markers
            // are visual-only, so skip them instead of throwing in Awake.
            if (SystemInfo.graphicsDeviceType == UnityEngine.Rendering.GraphicsDeviceType.Null)
            {
                return;
            }
            if (_markerMaterial == null)
            {
                Shader markerShader = Shader.Find("Sprites/Default");
                if (markerShader == null)
                {
                    return;
                }
                _markerMaterial = new Material(markerShader)
                {
                    color = new Color(1f, 1f, 1f, MARKER_ALPHA)
                };
            }

            int cellsX = Mathf.FloorToInt(width);
            int zStart = -4;
            int zEnd = Mathf.CeilToInt(length) + 2;
            int cellCount = 0;
            for (int cellZ = zStart; cellZ < zEnd; cellZ++)
            {
                for (int cellX = 0; cellX < cellsX; cellX++)
                {
                    if (((cellX + cellZ) & 1) == 0)
                    {
                        cellCount++;
                    }
                }
            }

            var vertices = new Vector3[cellCount * 4];
            var triangles = new int[cellCount * 6];
            int vertexIndex = 0;
            int triangleIndex = 0;
            float xOrigin = -width * 0.5f;
            for (int cellZ = zStart; cellZ < zEnd; cellZ++)
            {
                for (int cellX = 0; cellX < cellsX; cellX++)
                {
                    if (((cellX + cellZ) & 1) != 0)
                    {
                        continue;
                    }
                    float x0 = xOrigin + cellX;
                    float z0 = cellZ;
                    vertices[vertexIndex] = new Vector3(x0, SurfaceHeight(kind, x0, z0) + MARKER_HEIGHT_OFFSET, z0);
                    vertices[vertexIndex + 1] = new Vector3(x0, SurfaceHeight(kind, x0, z0 + 1f) + MARKER_HEIGHT_OFFSET, z0 + 1f);
                    vertices[vertexIndex + 2] = new Vector3(x0 + 1f, SurfaceHeight(kind, x0 + 1f, z0 + 1f) + MARKER_HEIGHT_OFFSET, z0 + 1f);
                    vertices[vertexIndex + 3] = new Vector3(x0 + 1f, SurfaceHeight(kind, x0 + 1f, z0) + MARKER_HEIGHT_OFFSET, z0);
                    triangles[triangleIndex++] = vertexIndex;
                    triangles[triangleIndex++] = vertexIndex + 1;
                    triangles[triangleIndex++] = vertexIndex + 2;
                    triangles[triangleIndex++] = vertexIndex;
                    triangles[triangleIndex++] = vertexIndex + 2;
                    triangles[triangleIndex++] = vertexIndex + 3;
                    vertexIndex += 4;
                }
            }
            var markerMesh = new Mesh { vertices = vertices, triangles = triangles };
            markerMesh.RecalculateNormals();

            var markers = new GameObject("ScaleMarkers");
            markers.transform.SetParent(parent, false);
            markers.AddComponent<MeshFilter>().sharedMesh = markerMesh;
            var markerRenderer = markers.AddComponent<MeshRenderer>();
            markerRenderer.sharedMaterial = _markerMaterial;
            markerRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            markerRenderer.receiveShadows = false;
        }

        /// <summary>
        /// Per-kind ground material: a clone of the base ground material with a
        /// procedural dirt/rock/bog texture. Cached so every rebuild and every
        /// training area shares one material per kind (one draw-call state).
        /// </summary>
        private Material GroundMaterialFor(TrackKind kind)
        {
            // Headless training builds strip rendering; texture work is wasted there.
            if (SystemInfo.graphicsDeviceType == UnityEngine.Rendering.GraphicsDeviceType.Null || _groundMaterial == null)
            {
                return _groundMaterial;
            }
            if (GroundMaterials.TryGetValue(kind, out Material cached) && cached != null)
            {
                return cached;
            }
            var material = new Material(_groundMaterial);
            // TrackGrid samples _BaseMap in world space; white base keeps the
            // texture's own colors, grid lines stay untouched.
            material.SetColor("_BaseColor", Color.white);
            material.SetTexture("_BaseMap", BuildGroundTexture(kind));
            GroundMaterials[kind] = material;
            return material;
        }

        private static Texture2D BuildGroundTexture(TrackKind kind)
        {
            Color baseColor;
            Color veinColor;
            if (kind == TrackKind.Swamp)
            {
                baseColor = new Color(0.24f, 0.3f, 0.18f);   // boggy moss
                veinColor = new Color(0.35f, 0.3f, 0.18f);   // drier peat
            }
            else if (kind == TrackKind.Lumpy || kind == TrackKind.Hills)
            {
                baseColor = new Color(0.45f, 0.34f, 0.22f);  // dry dirt
                veinColor = new Color(0.55f, 0.45f, 0.32f);
            }
            else if (kind == TrackKind.Rough || kind == TrackKind.RoughBlocked)
            {
                baseColor = new Color(0.38f, 0.37f, 0.36f);  // rock
                veinColor = new Color(0.5f, 0.48f, 0.45f);
            }
            else
            {
                baseColor = new Color(0.5f, 0.44f, 0.34f);   // packed sand
                veinColor = new Color(0.58f, 0.53f, 0.43f);
            }

            const int size = 256;
            var texture = new Texture2D(size, size, TextureFormat.RGB24, true)
            {
                wrapMode = TextureWrapMode.Repeat
            };
            var pixels = new Color[size * size];
            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    // Two Perlin octaves for blotches, one high-frequency speckle.
                    float broad = Mathf.PerlinNoise(x * 0.03f + (int)kind * 37f, y * 0.03f);
                    float detail = Mathf.PerlinNoise(x * 0.11f + 91f, y * 0.11f + 17f);
                    float speckle = Mathf.PerlinNoise(x * 0.45f + 5f, y * 0.45f + 43f);
                    float blend = Mathf.Clamp01(broad * 0.6f + detail * 0.4f);
                    Color color = Color.Lerp(baseColor, veinColor, blend);
                    float shade = 0.92f + speckle * 0.16f;
                    pixels[y * size + x] = new Color(color.r * shade, color.g * shade, color.b * shade);
                }
            }
            texture.SetPixels(pixels);
            texture.Apply(true);
            return texture;
        }

        /// <summary>
        /// Mud pits: a flat translucent brown quad on the ground plus a trigger
        /// box with a MudZoneView that drags every creature body inside.
        /// </summary>
        private static void BuildMudPits(Transform parent, float width, float length, System.Random rng, int count)
        {
            if (_mudMaterial == null && SystemInfo.graphicsDeviceType != UnityEngine.Rendering.GraphicsDeviceType.Null)
            {
                Shader mudShader = Shader.Find("Sprites/Default");
                if (mudShader != null)
                {
                    _mudMaterial = new Material(mudShader)
                    {
                        color = new Color(0.32f, 0.22f, 0.10f, 0.65f)
                    };
                }
            }

            for (int pitIndex = 0; pitIndex < count; pitIndex++)
            {
                float sizeX = 4f + (float)rng.NextDouble() * 3f;
                float sizeZ = 2.5f + (float)rng.NextDouble() * 1.5f;
                // Keep the first metres after the start clear so racers are not born in mud.
                float z = 4f + (float)rng.NextDouble() * (length - 6f);
                float x = ((float)rng.NextDouble() - 0.5f) * (width - sizeX);

                var pit = new GameObject("MudPit");
                pit.transform.SetParent(parent, false);
                pit.transform.localPosition = new Vector3(x, 0f, z);

                var trigger = pit.AddComponent<BoxCollider>();
                trigger.isTrigger = true;
                // Tall enough to catch every limb, sunk half below ground.
                trigger.size = new Vector3(sizeX, 1.2f, sizeZ);
                trigger.center = new Vector3(0f, 0.3f, 0f);
                pit.AddComponent<Views.MudZoneView>();

                if (_mudMaterial != null)
                {
                    var visual = GameObject.CreatePrimitive(PrimitiveType.Quad);
                    visual.name = "MudVisual";
                    UnityEngine.Object.Destroy(visual.GetComponent<Collider>());
                    visual.transform.SetParent(pit.transform, false);
                    visual.transform.localPosition = new Vector3(0f, MARKER_HEIGHT_OFFSET * 2f, 0f);
                    visual.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
                    visual.transform.localScale = new Vector3(sizeX, sizeZ, 1f);
                    var mudRenderer = visual.GetComponent<MeshRenderer>();
                    mudRenderer.sharedMaterial = _mudMaterial;
                    mudRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                    mudRenderer.receiveShadows = false;
                }
            }
        }

        /// <summary>
        /// Gates: a pair of low walls spanning the track with one random gap,
        /// so the field has to funnel or climb.
        /// </summary>
        private void BuildGates(Transform parent, float width, float length, System.Random rng, int count)
        {
            const float gapWidth = 9f;
            const float wallHeight = 0.35f;
            const float wallDepth = 0.4f;
            for (int gateIndex = 0; gateIndex < count; gateIndex++)
            {
                float z = length * (gateIndex + 1f) / (count + 1f);
                float gapCenter = ((float)rng.NextDouble() - 0.5f) * (width - gapWidth) * 0.5f;

                float leftEdge = -width * 0.5f;
                float leftWallWidth = (gapCenter - gapWidth * 0.5f) - leftEdge;
                float rightWallWidth = width * 0.5f - (gapCenter + gapWidth * 0.5f);

                if (leftWallWidth > 0.1f)
                {
                    BuildWall(parent, new Vector3(leftEdge + leftWallWidth * 0.5f, wallHeight * 0.5f, z),
                        new Vector3(leftWallWidth, wallHeight, wallDepth));
                }
                if (rightWallWidth > 0.1f)
                {
                    BuildWall(parent, new Vector3(width * 0.5f - rightWallWidth * 0.5f, wallHeight * 0.5f, z),
                        new Vector3(rightWallWidth, wallHeight, wallDepth));
                }
            }
        }

        private void BuildWall(Transform parent, Vector3 localPosition, Vector3 size)
        {
            var wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
            wall.name = "Gate";
            wall.transform.SetParent(parent, false);
            wall.transform.localPosition = localPosition;
            wall.transform.localScale = size;
            wall.GetComponent<MeshRenderer>().sharedMaterial = _obstacleMaterial;
            wall.GetComponent<BoxCollider>().sharedMaterial = _physicsMaterial;
        }

        /// <summary>
        /// Visual-only scenery: finish arch over the line, forest and tent tiles on
        /// the side aprons, bushes hugging the track edge. All colliders stripped so
        /// gameplay and physics are untouched; everything lives under 'parent' and
        /// is cleared with the track on rebuild.
        /// </summary>
        private static void DecorateTrack(TrackKind kind, Transform parent, float width, float length,
            float finishZ, System.Random rng)
        {
            if (SystemInfo.graphicsDeviceType == UnityEngine.Rendering.GraphicsDeviceType.Null)
            {
                return;
            }
            if (!_propsLoaded)
            {
                _propsLoaded = true;
                _bushPrefab = Resources.Load<GameObject>("Props/bush");
                _forestPrefab = Resources.Load<GameObject>("Props/deco_forest");
                _tentsPrefab = Resources.Load<GameObject>("Props/deco_tents");
                _archPrefab = Resources.Load<GameObject>("Props/finish_arch");

                Shader litShader = Shader.Find("Universal Render Pipeline/Lit");
                if (litShader != null)
                {
                    var atlas = Resources.Load<Texture2D>("Props/citybits_texture");
                    _propAtlasMaterial = new Material(litShader);
                    _propAtlasMaterial.SetFloat("_Smoothness", 0f);
                    if (atlas != null)
                    {
                        _propAtlasMaterial.SetTexture("_BaseMap", atlas);
                    }
                    _archMaterial = new Material(litShader);
                    _archMaterial.SetColor("_BaseColor", new Color(0.82f, 0.24f, 0.2f));
                    _archMaterial.SetFloat("_Smoothness", 0.25f);
                }
            }

            if (_archPrefab != null)
            {
                float archScale = width / ARCH_MODEL_WIDTH;
                PlaceProp(_archPrefab, parent,
                    new Vector3(0f, SurfaceHeight(kind, 0f, finishZ), finishZ),
                    rotationY: 0f, new Vector3(archScale, archScale * 0.8f, 1f), _archMaterial);
            }

            if (_forestPrefab != null)
            {
                // Forest tiles are 10 m squares; three per side, jittered so the
                // tree line never reads as a fence.
                for (int side = -1; side <= 1; side += 2)
                {
                    for (int tileIndex = 0; tileIndex < 3; tileIndex++)
                    {
                        float z = 1f + tileIndex * 11f + (float)rng.NextDouble() * 4f;
                        float x = side * (width * 0.5f + 8f + (float)rng.NextDouble() * 2.5f);
                        PlaceProp(_forestPrefab, parent,
                            new Vector3(x, SurfaceHeight(kind, x, z), z),
                            rotationY: rng.Next(4) * 90f, Vector3.one, _propAtlasMaterial);
                    }
                }
            }

            if (_tentsPrefab != null)
            {
                // Spectator camps flanking the start grid.
                for (int side = -1; side <= 1; side += 2)
                {
                    float x = side * (width * 0.5f + 8.5f);
                    float z = -4f + (float)rng.NextDouble() * 2f;
                    PlaceProp(_tentsPrefab, parent,
                        new Vector3(x, SurfaceHeight(kind, x, z), z),
                        rotationY: side < 0 ? 90f : 270f, Vector3.one, _propAtlasMaterial);
                }
            }

            if (_bushPrefab != null)
            {
                for (int bushIndex = 0; bushIndex < 22; bushIndex++)
                {
                    int side = rng.Next(2) == 0 ? -1 : 1;
                    float x = side * (width * 0.5f + 0.8f + (float)rng.NextDouble() * 2.8f);
                    float z = -4f + (float)rng.NextDouble() * (length + 8f);
                    float scale = 1.6f + (float)rng.NextDouble() * 1.8f;
                    PlaceProp(_bushPrefab, parent,
                        new Vector3(x, SurfaceHeight(kind, x, z), z),
                        rotationY: (float)rng.NextDouble() * 360f, Vector3.one * scale, overrideMaterial: null);
                }
            }
        }

        private static void PlaceProp(GameObject prefab, Transform parent, Vector3 localPosition,
            float rotationY, Vector3 scale, Material overrideMaterial)
        {
            GameObject prop = UnityEngine.Object.Instantiate(prefab, parent);
            prop.transform.localPosition = localPosition;
            prop.transform.localRotation = Quaternion.Euler(0f, rotationY, 0f);
            prop.transform.localScale = scale;
            Collider[] colliders = prop.GetComponentsInChildren<Collider>(true);
            for (int colliderIndex = 0; colliderIndex < colliders.Length; colliderIndex++)
            {
                UnityEngine.Object.Destroy(colliders[colliderIndex]);
            }
            if (overrideMaterial != null)
            {
                Renderer[] renderers = prop.GetComponentsInChildren<Renderer>(true);
                for (int rendererIndex = 0; rendererIndex < renderers.Length; rendererIndex++)
                {
                    renderers[rendererIndex].sharedMaterial = overrideMaterial;
                }
            }
        }

        private void ScatterBoxes(TrackKind kind, Transform parent, float width, float length, System.Random rng,
            int count, Vector3 minSize, Vector3 maxSize)
        {
            for (int boxIndex = 0; boxIndex < count; boxIndex++)
            {
                var box = GameObject.CreatePrimitive(PrimitiveType.Cube);
                box.name = "Obstacle";
                box.transform.SetParent(parent, false);
                var size = new Vector3(
                    Mathf.Lerp(minSize.x, maxSize.x, (float)rng.NextDouble()),
                    Mathf.Lerp(minSize.y, maxSize.y, (float)rng.NextDouble()),
                    Mathf.Lerp(minSize.z, maxSize.z, (float)rng.NextDouble()));
                box.transform.localScale = size;
                // Keep the first 3 m after the start line clear so racers are not born stuck.
                float z = 3f + (float)rng.NextDouble() * (length - 4f);
                float x = ((float)rng.NextDouble() - 0.5f) * (width - size.x);
                // Boxes sit on the local surface so rough tracks do not swallow them.
                box.transform.localPosition = new Vector3(x, SurfaceHeight(kind, x, z) + size.y * 0.5f, z);
                box.transform.localRotation = Quaternion.Euler(0f, (float)rng.NextDouble() * 40f - 20f, 0f);
                box.GetComponent<MeshRenderer>().sharedMaterial = _obstacleMaterial;
                box.GetComponent<BoxCollider>().sharedMaterial = _physicsMaterial;
            }
        }
    }
}
