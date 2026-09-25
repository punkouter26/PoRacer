using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// What <see cref="MujocoCreatureBuilder"/> hands back: the racer's root (destroy it to
    /// remove the racer) and the transforms the plug-in moves, for anything that must follow
    /// the body from outside MuJoCo (cross-simulator collision stand-ins).
    /// </summary>
    public sealed class MujocoCreatureInstance
    {
        public MujocoCreatureInstance(GameObject root, Transform[] bodies, Transform[] geoms)
        {
            Root = root;
            Bodies = bodies;
            Geoms = geoms;
        }

        public GameObject Root { get; }
        /// <summary>One per rig body, rig order.</summary>
        public Transform[] Bodies { get; }
        /// <summary>One per rig geom, rig order; each carries its shape along the plug-in's axes.</summary>
        public Transform[] Geoms { get; }
    }
}
