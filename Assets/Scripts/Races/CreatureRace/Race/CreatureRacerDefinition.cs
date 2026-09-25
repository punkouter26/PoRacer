using System;
using Unity.InferenceEngine;
using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// One lane of a creature race: who races there, how it was trained, what it looks like,
    /// its brain, and which simulator steps it inside Unity. The settings hold the list; the
    /// list index is the lane. The scene builders seed and refresh the known racers (names,
    /// colours, materials, brains) but never overwrite a physics choice made in the Inspector.
    ///
    /// Colour is a legend (AGENTS rule D): red and green are reserved and never used here.
    /// </summary>
    [Serializable]
    public sealed class CreatureRacerDefinition
    {
        private const string MUJOCO_PHYSICS_LABEL = "MuJoCo (org.mujoco plug-in)";
        private const string PHYSX_PHYSICS_LABEL = "PhysX ArticulationBody";

        [Tooltip("Shown on the HUD and in the results files.")]
        [SerializeField] private string _name = string.Empty;
        [Tooltip("Training method shown on the HUD, e.g. \"MuJoCo\", \"Isaac Lab 3\".")]
        [SerializeField] private string _method = string.Empty;
        [Tooltip("The simulator that steps this racer inside Unity. Pick the one closest to the "
               + "physics the brain trained in.")]
        [SerializeField] private CreaturePhysicsKind _physics = CreaturePhysicsKind.MujocoPlugin;
        [Tooltip("The exported ONNX, imported as a ModelAsset. Empty = the racer holds its rest pose (NO BRAIN).")]
        [SerializeField] private ModelAsset _brain;
        [Tooltip("File name expected in the creature's Brains folder, e.g. quad_mujoco.onnx. Used to find "
               + "the brain and to say exactly what to copy when it is missing.")]
        [SerializeField] private string _brainFile = string.Empty;
        [Tooltip("Shared body material (one per racer, so every part batches).")]
        [SerializeField] private Material _material;
        [Tooltip("HUD swatch; keep it the material's colour.")]
        [SerializeField] private Color _color = Color.white;

        public string Name => string.IsNullOrEmpty(_name) ? "racer" : _name;
        public string Method => _method ?? string.Empty;
        public CreaturePhysicsKind Physics => _physics;
        public ModelAsset Brain => _brain;
        public string BrainFile => _brainFile ?? string.Empty;
        public Material Material => _material;
        public Color Color => _color;

        public string PhysicsLabel => _physics == CreaturePhysicsKind.MujocoPlugin
            ? MUJOCO_PHYSICS_LABEL
            : PHYSX_PHYSICS_LABEL;
    }
}
