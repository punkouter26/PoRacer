using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// One geom's MuJoCo contact parameters: friction (sliding, torsional, rolling), condim,
    /// solref (time constant, damping ratio) and solimp (dmin, dmax, width). The plug-in's
    /// geom writer only carries these three solimp values, so a rig's midpoint and power
    /// (MuJoCo's defaults 0.5 and 2 in every rig so far) are not representable and ignored.
    /// </summary>
    public readonly struct CreatureContact
    {
        public CreatureContact(Vector3 friction, int conDim, Vector2 solRef, Vector3 solImp)
        {
            Friction = friction;
            ConDim = conDim;
            SolRef = solRef;
            SolImp = solImp;
        }

        /// <summary>MuJoCo's own geom defaults.</summary>
        public static CreatureContact MujocoDefault =>
            new(new Vector3(1f, 0.005f, 0.0001f), 3, new Vector2(0.02f, 1f), new Vector3(0.9f, 0.95f, 0.001f));

        /// <summary>x sliding, y torsional, z rolling.</summary>
        public Vector3 Friction { get; }
        public int ConDim { get; }
        /// <summary>x time constant, y damping ratio.</summary>
        public Vector2 SolRef { get; }
        /// <summary>x dmin, y dmax, z width.</summary>
        public Vector3 SolImp { get; }

        public CreatureContact WithSlidingFriction(float sliding)
        {
            return new CreatureContact(new Vector3(sliding, Friction.y, Friction.z), ConDim, SolRef, SolImp);
        }
    }
}
