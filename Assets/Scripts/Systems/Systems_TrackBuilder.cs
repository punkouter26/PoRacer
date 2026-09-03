using System;
using UnityEngine;

namespace PoRacer.Systems
{
    /// <summary>
    /// Optional hazards layered on top of a TrackKind. Swamp's built-in mud and
    /// gates stay implicit; these flags add extras to any kind (Gale map,
    /// Roulette rolls).
    /// </summary>
    [Flags]
    public enum TrackFeatures
    {
        None = 0,
        BoostPads = 1 << 0,
        Gusts = 1 << 1,
        MudPits = 1 << 2,
        Gates = 1 << 3
    }

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
        // High enough above the ground that decal quads never z-fight it.
        private const float MARKER_HEIGHT_OFFSET = 0.04f;
        // Flat-based props (forest/tent tiles) get the same lift for the same reason.
        private const float PROP_HEIGHT_OFFSET = 0.02f;
        // Visual-only side aprons that hold trees, tents, and bushes.
        private const float DECOR_MARGIN = 14f;
        private const float ARCH_MODEL_WIDTH = 14f;
        // Half-depth of the camera keep-out around the finish line. The arch's
        // solid band hugs the line; its renderer bounds are far deeper because of
        // the flat ground plate, which no camera can clip into.
        private const float ARCH_KEEPOUT_HALF_DEPTH = 1.6f;

        private static Material _mudMaterial;
        private static Material _mudRimMaterial;
        private static Material _boostMaterial;
        private static Material _boostRimMaterial;
        private const float BOOST_PAD_SIZE_X = 3.2f;
        private const float BOOST_PAD_SIZE_Z = 2.4f;
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

        // Scenery carries no colliders by design, so a physics-based camera
        // collision test cannot see the finish arch. The build publishes the
        // arch's world volume instead and Systems_Spawn hands it to the camera
        // director, which keeps the orbit shot out of it.
        private bool _hasFinishArchBounds;
        private Bounds _finishArchBounds;
        private bool _hasGroundBounds;
        private Bounds _groundBounds;

        /// <summary>
        /// World-space volume of the finish arch from the last <see cref="Build"/>,
        /// or false when the track was built without decoration.
        /// </summary>
        public bool TryGetFinishArchBounds(out Bounds bounds)
        {
            bounds = _finishArchBounds;
            return _hasFinishArchBounds;
        }

        /// <summary>
        /// World-space footprint of the ground the last Build laid down. RacerView
        /// bounds its runaway guard to this, so a racer that walks off the edge is
        /// caught at the edge instead of being held up over open space.
        /// </summary>
        public bool TryGetGroundBounds(out Bounds bounds)
        {
            bounds = _groundBounds;
            return _hasGroundBounds;
        }

        /// <summary>
        /// Adopts the footprint of a track that was authored in the scene instead
        /// of generated here, so the camera keep-out and the runaway guard read the
        /// same numbers either way.
        ///
        /// The alternative was to have the callers branch on which path produced
        /// the track. Feeding the bounds in keeps <see cref="TryGetGroundBounds"/>
        /// and <see cref="TryGetFinishArchBounds"/> the single source both
        /// consumers already read, so nothing downstream has to know.
        /// </summary>
        /// <summary>
        /// Destroys a component or object from either edit mode or play mode.
        ///
        /// Object.Destroy is deferred to the end of the frame and does nothing at
        /// all outside play mode. Every call here strips a collider that
        /// GameObject.CreatePrimitive added and that the track must not keep, so
        /// when this builder is run from the editor to bake a track, plain Destroy
        /// left the ground with its paper-thin plane MeshCollider still attached
        /// (racers sink straight through it) and left colliders on visual-only
        /// bleachers. DestroyImmediate is the edit-mode equivalent; the play-mode
        /// path is unchanged.
        /// </summary>
        private static void DestroyNow(UnityEngine.Object target)
        {
            if (target == null)
            {
                return;
            }
            if (Application.isPlaying)
            {
                UnityEngine.Object.Destroy(target);
            }
            else
            {
                UnityEngine.Object.DestroyImmediate(target);
            }
        }

        public void UseAuthored(Bounds groundBounds, bool hasArch, Bounds archBounds)
        {
            _groundBounds = groundBounds;
            _hasGroundBounds = true;
            _finishArchBounds = archBounds;
            _hasFinishArchBounds = hasArch;
        }

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
            bool decorate = false, float finishZ = -1f, TrackFeatures features = TrackFeatures.None,
            float backMargin = 7f)
        {
            for (int childIndex = parent.childCount - 1; childIndex >= 0; childIndex--)
            {
                DestroyNow(parent.GetChild(childIndex).gameObject);
            }

            float sideMargin = decorate ? DECOR_MARGIN : 0f;
            // Both ground builders lay the same footprint: full width across, and
            // z from -backMargin to length + 1 so deep start grids stay on it.
            float groundWidth = width + sideMargin * 2f;
            float groundDepth = length + 1f + backMargin;
            _groundBounds = new Bounds(
                parent.TransformPoint(new Vector3(0f, 0f, (length + 1f - backMargin) * 0.5f)),
                new Vector3(groundWidth, 1f, groundDepth));
            _hasGroundBounds = true;
            if (kind == TrackKind.Hills || kind == TrackKind.Rough || kind == TrackKind.RoughBlocked || kind == TrackKind.Lumpy)
            {
                BuildTerrainMesh(kind, parent, width, length, sideMargin, backMargin);
            }
            else
            {
                BuildFlatGround(kind, parent, width, length, sideMargin, backMargin);
            }

            if (kind == TrackKind.Bumps)
            {
                ScatterBoxes(kind, parent, width, length, rng, ScaleCount(24, length),
                    minSize: new Vector3(0.4f, 0.08f, 0.4f), maxSize: new Vector3(1.2f, 0.22f, 1.2f));
            }
            else if (kind == TrackKind.Walls || kind == TrackKind.RoughBlocked)
            {
                ScatterBoxes(kind, parent, width, length, rng, ScaleCount(10, length),
                    minSize: new Vector3(2f, 0.25f, 0.25f), maxSize: new Vector3(5f, 0.4f, 0.35f));
            }
            else if (kind == TrackKind.Lumpy)
            {
                ScatterBoxes(kind, parent, width, length, rng, ScaleCount(14, length),
                    minSize: new Vector3(0.5f, 0.3f, 0.5f), maxSize: new Vector3(1.8f, 0.9f, 1.8f));
            }
            else if (kind == TrackKind.Swamp)
            {
                BuildMudPits(parent, width, length, rng, ScaleCount(5, length));
                BuildGates(parent, width, length, rng, ScaleCount(2, length));
            }

            if ((features & TrackFeatures.MudPits) != 0 && kind != TrackKind.Swamp)
            {
                BuildMudPits(parent, width, length, rng, ScaleCount(3, length));
            }
            if ((features & TrackFeatures.Gates) != 0 && kind != TrackKind.Swamp)
            {
                BuildGates(parent, width, length, rng, ScaleCount(1, length));
            }
            if ((features & TrackFeatures.BoostPads) != 0)
            {
                BuildBoostPads(kind, parent, width, length, rng, ScaleCount(3, length));
            }
            if ((features & TrackFeatures.Gusts) != 0)
            {
                BuildGustZones(parent, width, length, rng, ScaleCount(2, length));
            }

            _hasFinishArchBounds = false;
            if (decorate)
            {
                DecorateTrack(kind, parent, width, length, finishZ > 0f ? finishZ : length - 2f, rng);
            }
        }

        /// <summary>
        /// Hazard/obstacle counts were authored for ~28 m tracks; longer maps
        /// (quadrupled 2026-08-17) keep the same density per meter. Never shrinks
        /// below the authored count, so training's 24 m areas are unchanged.
        /// </summary>
        private static int ScaleCount(int baseCount, float length)
        {
            return Mathf.Max(baseCount, Mathf.RoundToInt(baseCount * length / 28f));
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

        private void BuildFlatGround(TrackKind kind, Transform parent, float width, float length, float sideMargin,
            float backMargin)
        {
            float fullWidth = width + sideMargin * 2f;
            var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "Ground";
            ground.transform.SetParent(parent, false);
            // A plane primitive is 10x10 at scale 1; ground covers
            // z in [-backMargin, length + 1] so deep start grids stay on it.
            float depth = length + 1f + backMargin;
            ground.transform.localPosition = new Vector3(0f, 0f, (length + 1f - backMargin) * 0.5f);
            ground.transform.localScale = new Vector3(fullWidth / 10f, 1f, depth / 10f);
            ground.GetComponent<MeshRenderer>().sharedMaterial = GroundMaterialFor(kind);
            // The plane's MeshCollider is paper-thin: heavy limbs press through it
            // and creatures sink. Swap it for a solid slab whose top is at y = 0.
            DestroyNow(ground.GetComponent<MeshCollider>());
            var slab = ground.AddComponent<BoxCollider>();
            slab.size = new Vector3(10f, 1f, 10f);
            slab.center = new Vector3(0f, -0.5f, 0f);
            slab.sharedMaterial = _physicsMaterial;
        }

        private void BuildTerrainMesh(TrackKind kind, Transform parent, float width, float length, float sideMargin,
            float backMargin)
        {
            width += sideMargin * 2f;
            float fullLength = length + 3f + backMargin;
            float zStart = -backMargin;
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
            Texture2D groundTexture = BuildGroundTexture(kind);
            material.SetTexture("_BaseMap", groundTexture);
            // Bump derived from the same blotches so the dirt catches the sun.
            material.SetTexture("_BumpMap", BuildGroundNormalMap(groundTexture));
            material.EnableKeyword("_NORMALMAP");
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
        /// Tangent-space normal map from the ground texture's brightness: the
        /// blotches become shallow bumps, so the flat dirt reads as dirt.
        /// </summary>
        private static Texture2D BuildGroundNormalMap(Texture2D source)
        {
            const float strength = 1.6f;
            int size = source.width;
            var normalMap = new Texture2D(size, size, TextureFormat.RGB24, true, linear: true)
            {
                wrapMode = TextureWrapMode.Repeat
            };
            var pixels = new Color[size * size];
            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    float left = source.GetPixel((x - 1 + size) % size, y).grayscale;
                    float right = source.GetPixel((x + 1) % size, y).grayscale;
                    float down = source.GetPixel(x, (y - 1 + size) % size).grayscale;
                    float up = source.GetPixel(x, (y + 1) % size).grayscale;
                    var normal = new Vector3((left - right) * strength, (down - up) * strength, 1f).normalized;
                    pixels[y * size + x] = new Color(
                        normal.x * 0.5f + 0.5f, normal.y * 0.5f + 0.5f, normal.z * 0.5f + 0.5f);
                }
            }
            normalMap.SetPixels(pixels);
            normalMap.Apply(true);
            return normalMap;
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
                    // Dark, near-opaque goop plus a darker rim ring so pits read as
                    // a hazard from the race camera, not a faint stain.
                    _mudMaterial = new Material(mudShader)
                    {
                        color = new Color(0.26f, 0.17f, 0.07f, 0.88f)
                    };
                    _mudRimMaterial = new Material(mudShader)
                    {
                        color = new Color(0.12f, 0.08f, 0.03f, 0.9f)
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
                    BuildMudQuad(pit.transform, "MudRim", _mudRimMaterial,
                        new Vector3(sizeX * 1.14f, sizeZ * 1.14f, 1f), MARKER_HEIGHT_OFFSET);
                    BuildMudQuad(pit.transform, "MudVisual", _mudMaterial,
                        new Vector3(sizeX, sizeZ, 1f), MARKER_HEIGHT_OFFSET * 2f);
                }
            }
        }

        private static void BuildMudQuad(Transform parent, string name, Material material,
            Vector3 scale, float heightOffset)
        {
            var visual = GameObject.CreatePrimitive(PrimitiveType.Quad);
            visual.name = name;
            DestroyNow(visual.GetComponent<Collider>());
            visual.transform.SetParent(parent, false);
            visual.transform.localPosition = new Vector3(0f, heightOffset, 0f);
            visual.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
            visual.transform.localScale = scale;
            var mudRenderer = visual.GetComponent<MeshRenderer>();
            mudRenderer.sharedMaterial = material;
            mudRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            mudRenderer.receiveShadows = false;
        }

        /// <summary>
        /// Boost pads: a bright green quad on the ground plus a trigger box with a
        /// BoostPadView that pushes every creature body inside toward the finish.
        /// </summary>
        private static void BuildBoostPads(TrackKind kind, Transform parent, float width, float length,
            System.Random rng, int count)
        {
            if (_boostMaterial == null && SystemInfo.graphicsDeviceType != UnityEngine.Rendering.GraphicsDeviceType.Null)
            {
                Shader boostShader = Shader.Find("Sprites/Default");
                if (boostShader != null)
                {
                    _boostMaterial = new Material(boostShader)
                    {
                        color = new Color(0.15f, 0.85f, 0.45f, 0.75f)
                    };
                    _boostRimMaterial = new Material(boostShader)
                    {
                        color = new Color(0.05f, 0.45f, 0.22f, 0.85f)
                    };
                }
            }

            for (int padIndex = 0; padIndex < count; padIndex++)
            {
                // Spread pads down the track, one per segment, jittered inside it.
                float segment = (length - 9f) / count;
                float z = 5f + segment * padIndex + (float)rng.NextDouble() * segment * 0.7f;
                float x = ((float)rng.NextDouble() - 0.5f) * (width - BOOST_PAD_SIZE_X);

                var pad = new GameObject("BoostPad");
                pad.transform.SetParent(parent, false);
                pad.transform.localPosition = new Vector3(x, SurfaceHeight(kind, x, z), z);

                var trigger = pad.AddComponent<BoxCollider>();
                trigger.isTrigger = true;
                trigger.size = new Vector3(BOOST_PAD_SIZE_X, 1.2f, BOOST_PAD_SIZE_Z);
                trigger.center = new Vector3(0f, 0.3f, 0f);
                pad.AddComponent<Views.BoostPadView>();

                if (_boostMaterial != null)
                {
                    BuildMudQuad(pad.transform, "BoostRim", _boostRimMaterial,
                        new Vector3(BOOST_PAD_SIZE_X * 1.12f, BOOST_PAD_SIZE_Z * 1.12f, 1f), MARKER_HEIGHT_OFFSET);
                    BuildMudQuad(pad.transform, "BoostVisual", _boostMaterial,
                        new Vector3(BOOST_PAD_SIZE_X, BOOST_PAD_SIZE_Z, 1f), MARKER_HEIGHT_OFFSET * 2f);
                }
            }
        }

        /// <summary>
        /// Gust zones: full-width trigger stripes whose GustZoneView shoves the
        /// field sideways in waves. Direction and phase are rolled per zone.
        /// </summary>
        private static void BuildGustZones(Transform parent, float width, float length, System.Random rng, int count)
        {
            for (int zoneIndex = 0; zoneIndex < count; zoneIndex++)
            {
                // Keep the spawn pad calm; spread zones along the rest of the track.
                float z = 6f + (length - 8f) * (zoneIndex + 0.5f) / count
                    + ((float)rng.NextDouble() - 0.5f) * 3f;

                var zone = new GameObject("GustZone");
                zone.transform.SetParent(parent, false);
                zone.transform.localPosition = new Vector3(0f, 0f, z);

                var trigger = zone.AddComponent<BoxCollider>();
                trigger.isTrigger = true;
                trigger.size = new Vector3(width, 1.6f, 4f);
                trigger.center = new Vector3(0f, 0.5f, 0f);

                var view = zone.AddComponent<Views.GustZoneView>();
                view.Initialize(
                    rng.Next(2) == 0 ? -1f : 1f,
                    (float)rng.NextDouble() * 6f,
                    width);
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
        private void DecorateTrack(TrackKind kind, Transform parent, float width, float length,
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
                // The arch model has a 24x10 m base plate at exactly y = 0; lifted
                // slightly so it never z-fights the ground it sits on.
                GameObject arch = PlaceProp(_archPrefab, parent,
                    new Vector3(0f, SurfaceHeight(kind, 0f, finishZ) + PROP_HEIGHT_OFFSET + 0.01f, finishZ),
                    rotationY: 0f, new Vector3(archScale, archScale * 0.8f, 1f), _archMaterial);
                if (TryGetWorldBounds(arch, out Bounds archBounds))
                {
                    // The renderer bounds include the flat ground plate, which a
                    // camera at orbit height can never clip into; only the solid
                    // band around the line matters, so the depth is trimmed to it.
                    Vector3 extents = archBounds.extents;
                    extents.z = Mathf.Min(extents.z, ARCH_KEEPOUT_HALF_DEPTH);
                    Vector3 center = archBounds.center;
                    center.z = arch.transform.position.z;
                    _finishArchBounds = new Bounds(center, extents * 2f);
                    _hasFinishArchBounds = true;
                }
            }

            if (_forestPrefab != null)
            {
                // Forest tiles are 10 m squares, one row per ~11 m of track per
                // side, jittered so the tree line never reads as a fence.
                int forestTiles = Mathf.Max(3, Mathf.CeilToInt(length / 11f));
                for (int side = -1; side <= 1; side += 2)
                {
                    for (int tileIndex = 0; tileIndex < forestTiles; tileIndex++)
                    {
                        float z = 1f + tileIndex * 11f + (float)rng.NextDouble() * 4f;
                        float x = side * (width * 0.5f + 8f + (float)rng.NextDouble() * 2.5f);
                        PlaceProp(_forestPrefab, parent,
                            new Vector3(x, SurfaceHeight(kind, x, z) + PROP_HEIGHT_OFFSET, z),
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
                        new Vector3(x, SurfaceHeight(kind, x, z) + PROP_HEIGHT_OFFSET, z),
                        rotationY: side < 0 ? 90f : 270f, Vector3.one, _propAtlasMaterial);
                }
            }

            BuildCrowdStands(parent, width, length, rng);

            if (_bushPrefab != null)
            {
                int bushCount = ScaleCount(22, length);
                for (int bushIndex = 0; bushIndex < bushCount; bushIndex++)
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

        /// <summary>
        /// Grandstands: two per side, a dark bleacher wedge topped with a single
        /// vertex-colored crowd mesh that CrowdStandView bounces on big moments.
        /// Visual-only — no colliders.
        /// </summary>
        private static void BuildCrowdStands(Transform parent, float width, float length, System.Random rng)
        {
            const int standRows = 3;
            const int standColumns = 16;
            for (int side = -1; side <= 1; side += 2)
            {
                for (int standIndex = 0; standIndex < 2; standIndex++)
                {
                    float z = length * (0.28f + 0.36f * standIndex) + ((float)rng.NextDouble() - 0.5f) * 2f;
                    float x = side * (width * 0.5f + 3.2f);

                    var stand = new GameObject("CrowdStand");
                    stand.transform.SetParent(parent, false);
                    stand.transform.localPosition = new Vector3(x, 0f, z);
                    // Crowd mesh faces its local +z; turn each stand toward the track.
                    stand.transform.localRotation = Quaternion.Euler(0f, side < 0 ? 90f : -90f, 0f);

                    // Bleacher wedge: three rising steps, visual only.
                    for (int stepIndex = 0; stepIndex < standRows; stepIndex++)
                    {
                        var step = GameObject.CreatePrimitive(PrimitiveType.Cube);
                        step.name = "Bleacher";
                        DestroyNow(step.GetComponent<Collider>());
                        step.transform.SetParent(stand.transform, false);
                        step.transform.localPosition = new Vector3(0f, 0.2f + stepIndex * 0.45f, -0.8f * stepIndex);
                        step.transform.localScale = new Vector3(standColumns * 0.55f + 0.6f, 0.4f, 0.9f);
                        var stepRenderer = step.GetComponent<MeshRenderer>();
                        stepRenderer.sharedMaterial = _archMaterial != null ? _archMaterial : stepRenderer.sharedMaterial;
                        stepRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                    }

                    GameObject crowd = Views.CrowdStandView.BuildCrowdMesh(rng, standRows, standColumns);
                    crowd.transform.SetParent(stand.transform, false);
                    crowd.transform.localPosition = new Vector3(0f, 0.4f, 0f);
                    stand.AddComponent<Views.CrowdStandView>()
                        .Initialize(crowd.transform, (float)rng.NextDouble() * 3f);
                }
            }
        }

        private static GameObject PlaceProp(GameObject prefab, Transform parent, Vector3 localPosition,
            float rotationY, Vector3 scale, Material overrideMaterial)
        {
            GameObject prop = UnityEngine.Object.Instantiate(prefab, parent);
            prop.transform.localPosition = localPosition;
            prop.transform.localRotation = Quaternion.Euler(0f, rotationY, 0f);
            prop.transform.localScale = scale;
            Collider[] colliders = prop.GetComponentsInChildren<Collider>(true);
            for (int colliderIndex = 0; colliderIndex < colliders.Length; colliderIndex++)
            {
                DestroyNow(colliders[colliderIndex]);
            }
            if (overrideMaterial != null)
            {
                Renderer[] renderers = prop.GetComponentsInChildren<Renderer>(true);
                for (int rendererIndex = 0; rendererIndex < renderers.Length; rendererIndex++)
                {
                    renderers[rendererIndex].sharedMaterial = overrideMaterial;
                }
            }
            return prop;
        }

        /// <summary>
        /// World bounds of every renderer under 'prop', or false when it has none.
        /// </summary>
        private static bool TryGetWorldBounds(GameObject prop, out Bounds bounds)
        {
            bounds = default;
            Renderer[] renderers = prop.GetComponentsInChildren<Renderer>(true);
            if (renderers.Length == 0)
            {
                return false;
            }
            bounds = renderers[0].bounds;
            for (int rendererIndex = 1; rendererIndex < renderers.Length; rendererIndex++)
            {
                bounds.Encapsulate(renderers[rendererIndex].bounds);
            }
            return true;
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
