using System;
using UnityEngine;

namespace PoRacer.Views
{
    public enum CosmeticType
    {
        TopHat,
        RoyalCrown,
        PartyCone,
        PropellerBeanie,
        Sombrero,
        Jetpack,
        VikingHorns
    }

    /// <summary>
    /// Silly physics-driven cosmetic prop attached to a racer's head/root.
    /// Features procedural geometry, spinning propellers/accessories, leader speed
    /// particle auras, and a dramatic physics pop-off ejection when the racer wipes out.
    /// </summary>
    public sealed class CosmeticPropView : MonoBehaviour
    {
        private const float POP_OFF_UPWARD_FORCE = 4.5f;
        private const float POP_OFF_HORIZONTAL_FORCE = 2.5f;
        private const float POP_OFF_TORQUE = 15f;
        private const float LEADER_AURA_SPEED_THRESHOLD = 1.2f;

        private CosmeticType _type;
        private GameObject _propObject;
        private Transform _propTransform;
        private Transform _propellerBlade;
        private ParticleSystem _auraParticles;
        private bool _isEjected;
        private bool _isLeader;
        private Rigidbody _propRigidbody;
        private Vector3 _lastPosition;
        private float _speed;

        public bool IsEjected => _isEjected;

        public void Initialize(CosmeticType type, Color baseTint, Transform targetAnchor = null)
        {
            _type = type;
            Transform anchor = targetAnchor != null ? targetAnchor : FindBestAnchor();

            _propObject = new GameObject($"Cosmetic_{type}");
            _propTransform = _propObject.transform;
            _propTransform.SetParent(anchor, false);

            BuildPropGeometry(type, baseTint);
            BuildLeaderAura(baseTint);

            _lastPosition = transform.position;
        }

        public void SetLeader(bool isLeader)
        {
            _isLeader = isLeader;
        }

        private void Update()
        {
            if (_isEjected)
            {
                return;
            }

            Vector3 currentPos = transform.position;
            _speed = (currentPos - _lastPosition).magnitude / Mathf.Max(Time.deltaTime, 0.001f);
            _lastPosition = currentPos;

            // Spin propeller if beanie
            if (_propellerBlade != null)
            {
                _propellerBlade.Rotate(Vector3.up, 720f * Time.deltaTime, Space.Self);
            }

            // Update leader / high-speed aura
            if (_auraParticles != null)
            {
                var emission = _auraParticles.emission;
                bool active = _isLeader || _speed > LEADER_AURA_SPEED_THRESHOLD;
                emission.enabled = active;
            }
        }

        /// <summary>
        /// Pop off the cosmetic prop with explosive physics ejection when the racer
        /// suffers a collision, knockdown, or wipeout.
        /// </summary>
        public void Eject()
        {
            if (_isEjected || _propObject == null)
            {
                return;
            }

            _isEjected = true;
            _propTransform.SetParent(null, true);

            if (_auraParticles != null)
            {
                var emission = _auraParticles.emission;
                emission.enabled = false;
            }

            // Add physics components for tumbling
            _propRigidbody = _propObject.AddComponent<Rigidbody>();
            _propRigidbody.mass = 0.4f;
            _propRigidbody.interpolation = RigidbodyInterpolation.Interpolate;

            BoxCollider collider = _propObject.AddComponent<BoxCollider>();
            collider.size = new Vector3(0.4f, 0.4f, 0.4f);

            // Explosive impulse
            Vector3 randomDir = UnityEngine.Random.insideUnitSphere;
            randomDir.y = Mathf.Abs(randomDir.y);
            Vector3 impulse = (Vector3.up * POP_OFF_UPWARD_FORCE) + (randomDir * POP_OFF_HORIZONTAL_FORCE);
            _propRigidbody.AddForce(impulse, ForceMode.Impulse);

            Vector3 torque = new Vector3(
                UnityEngine.Random.Range(-POP_OFF_TORQUE, POP_OFF_TORQUE),
                UnityEngine.Random.Range(-POP_OFF_TORQUE, POP_OFF_TORQUE),
                UnityEngine.Random.Range(-POP_OFF_TORQUE, POP_OFF_TORQUE)
            );
            _propRigidbody.AddTorque(torque, ForceMode.Impulse);

            FxUtil.KnockoutPuff(_propTransform.position);
            Destroy(_propObject, 8f);
        }

        private Transform FindBestAnchor()
        {
            // Look for Head or uppermost bone
            Transform[] children = GetComponentsInChildren<Transform>();
            Transform best = null;
            float highestY = float.MinValue;

            for (int i = 0; i < children.Length; i++)
            {
                Transform child = children[i];
                string lower = child.name.ToLowerInvariant();
                if (lower.Contains("head"))
                {
                    return child;
                }
                if (child.position.y > highestY)
                {
                    highestY = child.position.y;
                    best = child;
                }
            }

            return best != null ? best : transform;
        }

        private void BuildPropGeometry(CosmeticType type, Color tint)
        {
            GameObject visual = new GameObject("Visual");
            visual.transform.SetParent(_propTransform, false);

            MeshFilter mf = visual.AddComponent<MeshFilter>();
            MeshRenderer mr = visual.AddComponent<MeshRenderer>();

            Material mat = new Material(Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard") ?? Shader.Find("Sprites/Default"));
            mr.material = mat;

            switch (type)
            {
                case CosmeticType.TopHat:
                    mf.sharedMesh = CreateCylinderMesh(0.18f, 0.28f, 16);
                    visual.transform.localPosition = new Vector3(0f, 0.28f, 0f);
                    mat.color = new Color(0.12f, 0.12f, 0.14f);

                    // Add Brim
                    GameObject brim = new GameObject("Brim");
                    brim.transform.SetParent(_propTransform, false);
                    brim.transform.localPosition = new Vector3(0f, 0.15f, 0f);
                    MeshFilter bmf = brim.AddComponent<MeshFilter>();
                    MeshRenderer bmr = brim.AddComponent<MeshRenderer>();
                    bmf.sharedMesh = CreateCylinderMesh(0.35f, 0.03f, 16);
                    bmr.material = mat;
                    break;

                case CosmeticType.RoyalCrown:
                    mf.sharedMesh = CreateCrownMesh(0.24f, 0.22f, 8);
                    visual.transform.localPosition = new Vector3(0f, 0.25f, 0f);
                    mat.color = new Color(1f, 0.85f, 0.1f);
                    break;

                case CosmeticType.PartyCone:
                    mf.sharedMesh = CreateConeMesh(0.18f, 0.35f, 12);
                    visual.transform.localPosition = new Vector3(0f, 0.25f, 0f);
                    mat.color = tint;
                    break;

                case CosmeticType.PropellerBeanie:
                    mf.sharedMesh = CreateCylinderMesh(0.22f, 0.14f, 12);
                    visual.transform.localPosition = new Vector3(0f, 0.20f, 0f);
                    mat.color = new Color(0.2f, 0.5f, 0.9f);

                    // Propeller Blade
                    GameObject blade = new GameObject("Propeller");
                    blade.transform.SetParent(_propTransform, false);
                    blade.transform.localPosition = new Vector3(0f, 0.36f, 0f);
                    MeshFilter bladeMf = blade.AddComponent<MeshFilter>();
                    MeshRenderer bladeMr = blade.AddComponent<MeshRenderer>();
                    bladeMf.sharedMesh = CreateBoxMesh(0.36f, 0.02f, 0.06f);
                    Material bladeMat = new Material(mat);
                    bladeMat.color = new Color(1f, 0.8f, 0.1f);
                    bladeMr.material = bladeMat;
                    _propellerBlade = blade.transform;
                    break;

                case CosmeticType.Sombrero:
                    mf.sharedMesh = CreateConeMesh(0.16f, 0.22f, 12);
                    visual.transform.localPosition = new Vector3(0f, 0.25f, 0f);
                    mat.color = new Color(0.85f, 0.72f, 0.45f);

                    // Wide Brim
                    GameObject sBrim = new GameObject("SombreroBrim");
                    sBrim.transform.SetParent(_propTransform, false);
                    sBrim.transform.localPosition = new Vector3(0f, 0.16f, 0f);
                    MeshFilter sbmf = sBrim.AddComponent<MeshFilter>();
                    MeshRenderer sbmr = sBrim.AddComponent<MeshRenderer>();
                    sbmf.sharedMesh = CreateCylinderMesh(0.48f, 0.04f, 16);
                    sbmr.material = mat;
                    break;

                case CosmeticType.Jetpack:
                    mf.sharedMesh = CreateCylinderMesh(0.10f, 0.38f, 10);
                    visual.transform.localPosition = new Vector3(-0.14f, 0.1f, -0.2f);
                    mat.color = new Color(0.7f, 0.2f, 0.15f);

                    // Twin canister
                    GameObject jet2 = new GameObject("Jet2");
                    jet2.transform.SetParent(_propTransform, false);
                    jet2.transform.localPosition = new Vector3(0.14f, 0.1f, -0.2f);
                    MeshFilter j2mf = jet2.AddComponent<MeshFilter>();
                    MeshRenderer j2mr = jet2.AddComponent<MeshRenderer>();
                    j2mf.sharedMesh = mf.sharedMesh;
                    j2mr.material = mat;
                    break;

                case CosmeticType.VikingHorns:
                    mf.sharedMesh = CreateCylinderMesh(0.24f, 0.08f, 12);
                    visual.transform.localPosition = new Vector3(0f, 0.2f, 0f);
                    mat.color = new Color(0.35f, 0.25f, 0.18f);

                    // Left Horn
                    GameObject hornL = new GameObject("HornL");
                    hornL.transform.SetParent(_propTransform, false);
                    hornL.transform.localPosition = new Vector3(-0.2f, 0.28f, 0f);
                    hornL.transform.localRotation = Quaternion.Euler(0f, 0f, 35f);
                    MeshFilter hlmf = hornL.AddComponent<MeshFilter>();
                    MeshRenderer hlmr = hornL.AddComponent<MeshRenderer>();
                    hlmf.sharedMesh = CreateConeMesh(0.06f, 0.25f, 8);
                    Material hornMat = new Material(mat);
                    hornMat.color = new Color(0.95f, 0.95f, 0.9f);
                    hlmr.material = hornMat;

                    // Right Horn
                    GameObject hornR = new GameObject("HornR");
                    hornR.transform.SetParent(_propTransform, false);
                    hornR.transform.localPosition = new Vector3(0.2f, 0.28f, 0f);
                    hornR.transform.localRotation = Quaternion.Euler(0f, 0f, -35f);
                    MeshFilter hrmf = hornR.AddComponent<MeshFilter>();
                    MeshRenderer hrmr = hornR.AddComponent<MeshRenderer>();
                    hrmf.sharedMesh = hlmf.sharedMesh;
                    hrmr.material = hornMat;
                    break;
            }
        }

        private void BuildLeaderAura(Color tint)
        {
            GameObject auraObj = new GameObject("LeaderAura");
            auraObj.transform.SetParent(_propTransform, false);
            auraObj.transform.localPosition = Vector3.zero;

            _auraParticles = auraObj.AddComponent<ParticleSystem>();
            var renderer = auraObj.GetComponent<ParticleSystemRenderer>();
            Material glowMat = FxUtil.GlowParticleMaterial();
            if (glowMat != null)
            {
                renderer.material = glowMat;
            }

            var main = _auraParticles.main;
            main.startSize = 0.2f;
            main.startLifetime = 0.5f;
            main.startSpeed = 0.8f;
            main.startColor = new Color(1f, 0.85f, 0.2f, 0.7f);
            main.simulationSpace = ParticleSystemSimulationSpace.World;

            var emission = _auraParticles.emission;
            emission.rateOverTime = 18f;
            emission.enabled = false;

            var shape = _auraParticles.shape;
            shape.shapeType = ParticleSystemShapeType.Sphere;
            shape.radius = 0.3f;
        }

        private static Mesh CreateCylinderMesh(float radius, float height, int segments)
        {
            Mesh mesh = new Mesh { name = "ProcCylinder" };
            int vertexCount = (segments + 1) * 2;
            Vector3[] vertices = new Vector3[vertexCount];
            int[] triangles = new int[segments * 6];

            float halfH = height * 0.5f;
            for (int i = 0; i <= segments; i++)
            {
                float angle = (float)i / segments * Mathf.PI * 2f;
                float x = Mathf.Cos(angle) * radius;
                float z = Mathf.Sin(angle) * radius;

                vertices[i] = new Vector3(x, halfH, z);
                vertices[i + segments + 1] = new Vector3(x, -halfH, z);

                if (i < segments)
                {
                    int t = i * 6;
                    int b = i + segments + 1;
                    triangles[t] = i;
                    triangles[t + 1] = i + 1;
                    triangles[t + 2] = b;

                    triangles[t + 3] = i + 1;
                    triangles[t + 4] = b + 1;
                    triangles[t + 5] = b;
                }
            }

            mesh.vertices = vertices;
            mesh.triangles = triangles;
            mesh.RecalculateNormals();
            return mesh;
        }

        private static Mesh CreateConeMesh(float radius, float height, int segments)
        {
            Mesh mesh = new Mesh { name = "ProcCone" };
            Vector3[] vertices = new Vector3[segments + 2];
            int[] triangles = new int[segments * 3];

            vertices[0] = new Vector3(0f, height, 0f);
            for (int i = 0; i <= segments; i++)
            {
                float angle = (float)i / segments * Mathf.PI * 2f;
                vertices[i + 1] = new Vector3(Mathf.Cos(angle) * radius, 0f, Mathf.Sin(angle) * radius);

                if (i < segments)
                {
                    int t = i * 3;
                    triangles[t] = 0;
                    triangles[t + 1] = i + 1;
                    triangles[t + 2] = i + 2;
                }
            }

            mesh.vertices = vertices;
            mesh.triangles = triangles;
            mesh.RecalculateNormals();
            return mesh;
        }

        private static Mesh CreateCrownMesh(float radius, float height, int spikes)
        {
            Mesh mesh = new Mesh { name = "ProcCrown" };
            int count = spikes * 2;
            Vector3[] vertices = new Vector3[count + 1];
            int[] triangles = new int[count * 3];

            vertices[0] = Vector3.zero;
            for (int i = 0; i < count; i++)
            {
                float angle = (float)i / count * Mathf.PI * 2f;
                float h = (i % 2 == 0) ? height : height * 0.4f;
                vertices[i + 1] = new Vector3(Mathf.Cos(angle) * radius, h, Mathf.Sin(angle) * radius);

                int t = i * 3;
                triangles[t] = 0;
                triangles[t + 1] = i + 1;
                triangles[t + 2] = (i + 1) % count + 1;
            }

            mesh.vertices = vertices;
            mesh.triangles = triangles;
            mesh.RecalculateNormals();
            return mesh;
        }

        private static Mesh CreateBoxMesh(float width, float height, float depth)
        {
            Mesh mesh = new Mesh { name = "ProcBox" };
            float hx = width * 0.5f;
            float hy = height * 0.5f;
            float hz = depth * 0.5f;

            Vector3[] vertices =
            {
                new(-hx, -hy, -hz), new(hx, -hy, -hz), new(hx, hy, -hz), new(-hx, hy, -hz),
                new(-hx, -hy, hz), new(hx, -hy, hz), new(hx, hy, hz), new(-hx, hy, hz)
            };

            int[] triangles =
            {
                0, 2, 1, 0, 3, 2,
                4, 5, 6, 4, 6, 7,
                0, 1, 5, 0, 5, 4,
                2, 3, 7, 2, 7, 6,
                0, 4, 7, 0, 7, 3,
                1, 2, 6, 1, 6, 5
            };

            mesh.vertices = vertices;
            mesh.triangles = triangles;
            mesh.RecalculateNormals();
            return mesh;
        }
    }
}
