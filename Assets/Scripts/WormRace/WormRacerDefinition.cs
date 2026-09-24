using System;
using Unity.InferenceEngine;
using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// One lane of the worm race: who races there, how it was trained, what it looks like,
    /// its brain, and which simulator steps it inside Unity. WormRaceSettings holds the list;
    /// the list index is the lane. Editor_BuildWormRaceScene seeds and refreshes the known
    /// racers (names, colours, materials, brains) but never overwrites a physics choice made
    /// in the Inspector.
    ///
    /// Colour is a legend (AGENTS rule D): red and green are reserved and never used here.
    /// </summary>
    [Serializable]
    internal sealed class WormRacerDefinition
    {
        private const string MUJOCO_PHYSICS_LABEL = "MuJoCo (org.mujoco plug-in)";
        private const string PHYSX_PHYSICS_LABEL = "PhysX ArticulationBody";

        [Tooltip("Shown on the HUD and in the results files.")]
        [SerializeField] private string _name = string.Empty;
        [Tooltip("Training method shown on the HUD, e.g. \"MuJoCo\", \"Isaac Lab\", \"Isaac Lab 3\".")]
        [SerializeField] private string _method = string.Empty;
        [Tooltip("The simulator that steps this worm inside Unity. Pick the one closest to the "
               + "physics the brain trained in.")]
        [SerializeField] private WormPhysicsKind _physics = WormPhysicsKind.MujocoPlugin;
        [Tooltip("The exported ONNX, imported as a ModelAsset. Empty = the worm lies still (NO BRAIN).")]
        [SerializeField] private ModelAsset _brain;
        [Tooltip("File name expected under Assets/WormRace/Brains/, e.g. worm_mujoco.onnx. Used to find "
               + "the brain and to say exactly what to copy when it is missing.")]
        [SerializeField] private string _brainFile = string.Empty;
        [Tooltip("Shared segment material (one per racer, so all five segments batch).")]
        [SerializeField] private Material _material;
        [Tooltip("HUD swatch; keep it the material's colour.")]
        [SerializeField] private Color _color = Color.white;

        internal string Name => string.IsNullOrEmpty(_name) ? "worm" : _name;
        internal string Method => _method ?? string.Empty;
        internal WormPhysicsKind Physics => _physics;
        internal ModelAsset Brain => _brain;
        internal Material Material => _material;
        internal Color Color => _color;

        /// <summary>Where the brain is expected, for the missing-brain message.</summary>
        internal string ExpectedBrainPath => WormRacePaths.BRAINS + "/"
            + (string.IsNullOrEmpty(_brainFile) ? "<brain>.onnx" : _brainFile);

        internal string PhysicsLabel => _physics == WormPhysicsKind.MujocoPlugin
            ? MUJOCO_PHYSICS_LABEL
            : PHYSX_PHYSICS_LABEL;
    }
}
