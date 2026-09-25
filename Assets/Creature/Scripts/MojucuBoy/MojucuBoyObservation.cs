using System;
using Mujoco;
using UnityEngine;

namespace Creature.MojucuBoy
{
    /// <summary>
    /// The single source of truth for MojucuBoy's 75-element observation vector.
    ///
    /// This is a CONTRACT with the Python side: training/mojucuboy/mojucuboy_env.py builds the
    /// same vector in the same order. Change one, change both, or the policy is fed
    /// garbage that looks plausible.
    ///
    /// <code>
    ///   0.. 2  gravity_local  R^T * (0,0,-1), R = xmat of the root body
    ///   3.. 5  linvel_local   R^T * qvel[0:3]  (a free joint stores linear velocity in WORLD)
    ///   6.. 8  angvel_local   qvel[3:6]        (a free joint stores angular velocity in BODY)
    ///   9..11  command        cos(heading error), sin(heading error), target speed
    ///  12..32  joint_pos      qpos, in actuator order
    ///  33..53  joint_vel      qvel, in actuator order
    ///  54..74  last_action    previous clipped action
    /// </code>
    ///
    /// Everything is read straight out of MuJoCo's own mjData, in MuJoCo's frame --
    /// deliberately, and the same way CreatureAgent does it for Fido. None of
    /// MjEngineTool's Unity conversions touch the observation, so there is no frame
    /// mapping here that could drift away from what the policy trained against.
    ///
    /// Forward is body-local +Y, because the rig is authored facing MuJoCo +Y so that
    /// org.mujoco maps it onto Unity +Z, the race direction. See
    /// training/mojucuboy/build_mjcf.py.
    /// </summary>
    public static class MojucuBoyObservation
    {
        public const int JOINT_COUNT = 21;
        public const int OBS_SIZE = 9 + 3 + 3 * JOINT_COUNT;   // 75
        /// <summary>The joystick layout: command vx/vy/wz, then cos/sin of the gait clock.
        /// Contract with training/mojucuboy/mojucuboy_joystick_env.py.</summary>
        public const int JOYSTICK_OBS_SIZE = 9 + 3 + 2 + 3 * JOINT_COUNT;   // 77
        public const int ACTION_SIZE = JOINT_COUNT;

        /// <summary>Index of the body-local forward axis. +Y, see the class remarks.</summary>
        public const int FORWARD_AXIS = 1;

        /// <summary>
        /// Fill <paramref name="obs"/> from the live MuJoCo state.
        /// </summary>
        /// <param name="data">MuJoCo data for the stepped scene.</param>
        /// <param name="rootBodyId">MuJoCo body id of the free-joint root.</param>
        /// <param name="rootDofAddr">qvel address of THIS racer's free joint.</param>
        /// <param name="qposAddr">qpos address of each actuated joint, in actuator order.</param>
        /// <param name="dofAddr">qvel address of each actuated joint, in actuator order.</param>
        /// <param name="commandHeading">Commanded world heading, radians, measured the
        /// same way as the racer's own heading.</param>
        /// <param name="commandSpeed">Commanded forward speed, m/s.</param>
        /// <param name="lastAction">The previous clipped action.</param>
        /// <param name="obs">Destination, length <see cref="OBS_SIZE"/>.</param>
        public static unsafe void Build(
            MujocoLib.mjData_* data,
            int rootBodyId,
            int rootDofAddr,
            int[] qposAddr,
            int[] dofAddr,
            float commandHeading,
            float commandSpeed,
            float[] lastAction,
            float[] obs)
        {
            if (obs == null || obs.Length != OBS_SIZE)
            {
                throw new ArgumentException($"obs must be length {OBS_SIZE}", nameof(obs));
            }

            double* xmat = data->xmat + 9 * rootBodyId;
            BuildBody(data, xmat, rootDofAddr, obs);

            // command: heading error expressed as a unit vector so the policy never sees
            // the +pi/-pi discontinuity, plus the requested speed.
            float error = HeadingError(xmat, commandHeading);
            obs[9] = Mathf.Cos(error);
            obs[10] = Mathf.Sin(error);
            obs[11] = commandSpeed;

            BuildJoints(data, qposAddr, dofAddr, lastAction, obs, 12);
        }

        /// <summary>
        /// Fill <paramref name="obs"/> in the joystick layout (<see cref="JOYSTICK_OBS_SIZE"/>):
        /// the body block, then the body-frame velocity command (forward vx, left vy, yaw rate
        /// wz), then cos/sin of the LEFT foot's gait phase, then the joint block.
        /// </summary>
        public static unsafe void BuildJoystick(
            MujocoLib.mjData_* data,
            int rootBodyId,
            int rootDofAddr,
            int[] qposAddr,
            int[] dofAddr,
            Vector3 command,
            float phase,
            float[] lastAction,
            float[] obs)
        {
            if (obs == null || obs.Length != JOYSTICK_OBS_SIZE)
            {
                throw new ArgumentException($"obs must be length {JOYSTICK_OBS_SIZE}", nameof(obs));
            }
            BuildBody(data, data->xmat + 9 * rootBodyId, rootDofAddr, obs);
            obs[9] = command.x;
            obs[10] = command.y;
            obs[11] = command.z;
            obs[12] = Mathf.Cos(phase);
            obs[13] = Mathf.Sin(phase);
            BuildJoints(data, qposAddr, dofAddr, lastAction, obs, 14);
        }

        /// <summary>
        /// Gravity, linear and angular velocity in the body frame, into obs[0..8].
        ///
        /// Velocities are read at THIS racer's free-joint address. They were read from
        /// qvel[0..5], which is the free joint of whichever racer compiled first: with one
        /// MojucuBoy that is him, but with five on the grid, four of them balanced and
        /// steered on the first one's motion.
        /// </summary>
        private static unsafe void BuildBody(MujocoLib.mjData_* data, double* xmat, int rootDofAddr, float[] obs)
        {
            // xmat is row-major 3x3 per body: R[r][c] = xmat[3*r + c], body -> world.
            double* qvel = data->qvel + rootDofAddr;

            // gravity_local = R^T * (0,0,-1)  ->  -R[2][c]
            obs[0] = (float)(-xmat[6]);
            obs[1] = (float)(-xmat[7]);
            obs[2] = (float)(-xmat[8]);

            // linvel_local = R^T * qvel[0:3]  ->  sum over r of R[r][c] * qvel[r]
            for (int c = 0; c < 3; c++)
            {
                obs[3 + c] = (float)(xmat[c] * qvel[0] + xmat[3 + c] * qvel[1] + xmat[6 + c] * qvel[2]);
            }

            // angvel_local: a MuJoCo free joint already stores this in the body frame.
            obs[6] = (float)qvel[3];
            obs[7] = (float)qvel[4];
            obs[8] = (float)qvel[5];
        }

        private static unsafe void BuildJoints(MujocoLib.mjData_* data, int[] qposAddr, int[] dofAddr,
                                               float[] lastAction, float[] obs, int start)
        {
            double* qpos = data->qpos;
            double* qvel = data->qvel;
            for (int i = 0; i < JOINT_COUNT; i++)
            {
                obs[start + i] = (float)qpos[qposAddr[i]];
                obs[start + JOINT_COUNT + i] = (float)qvel[dofAddr[i]];
            }
            Array.Copy(lastAction, 0, obs, start + 2 * JOINT_COUNT, JOINT_COUNT);
        }

        /// <summary>
        /// Signed angle from the racer's current world heading to the commanded one,
        /// wrapped to (-pi, pi]. Heading is the body's forward axis flattened onto the
        /// world XY plane.
        /// </summary>
        public static unsafe float HeadingError(double* xmat, float commandHeading)
        {
            // forward_world = R * e_FORWARD  ->  component r is R[r][FORWARD_AXIS]
            double fx = xmat[FORWARD_AXIS];
            double fy = xmat[3 + FORWARD_AXIS];
            float heading = (float)Math.Atan2(fy, fx);
            float error = commandHeading - heading;
            while (error > Mathf.PI) { error -= 2f * Mathf.PI; }
            while (error <= -Mathf.PI) { error += 2f * Mathf.PI; }
            return error;
        }
    }
}
