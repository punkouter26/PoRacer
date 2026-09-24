using System;
using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// The one place the 35-float observation is assembled, for both worms. Each view
    /// converts its own physics state into a <see cref="WormBodyState"/>; from there on the
    /// MuJoCo worm and the PhysX worm go through identical code, so a difference in how
    /// they race cannot come from a difference in what they see.
    /// </summary>
    internal static class WormObservation
    {
        private const float MIN_GOAL_NORM = 1e-6f;

        public static void Build(in WormBodyState body, float[] jointPositions, float[] jointVelocities,
                                 float[] previousAction, float[] obs)
        {
            if (obs == null || obs.Length != WormContract.OBS_SIZE)
            {
                throw new ArgumentException($"obs must be length {WormContract.OBS_SIZE}", nameof(obs));
            }

            // Gravity (0, 0, -1) in B: R^T g, i.e. minus the z component of each B axis.
            obs[WormContract.OBS_GRAVITY] = -body.AxisX.z;
            obs[WormContract.OBS_GRAVITY + 1] = -body.AxisY.z;
            obs[WormContract.OBS_GRAVITY + 2] = -body.AxisZ.z;

            Vector3 velocity = body.LinearVelocity;
            obs[WormContract.OBS_LINEAR_VELOCITY] = Vector3.Dot(velocity, body.AxisX) * WormContract.LINEAR_VELOCITY_SCALE;
            obs[WormContract.OBS_LINEAR_VELOCITY + 1] = Vector3.Dot(velocity, body.AxisY) * WormContract.LINEAR_VELOCITY_SCALE;
            obs[WormContract.OBS_LINEAR_VELOCITY + 2] = Vector3.Dot(velocity, body.AxisZ) * WormContract.LINEAR_VELOCITY_SCALE;

            Vector3 spin = body.AngularVelocity;
            obs[WormContract.OBS_ANGULAR_VELOCITY] = Vector3.Dot(spin, body.AxisX) * WormContract.ANGULAR_VELOCITY_SCALE;
            obs[WormContract.OBS_ANGULAR_VELOCITY + 1] = Vector3.Dot(spin, body.AxisY) * WormContract.ANGULAR_VELOCITY_SCALE;
            obs[WormContract.OBS_ANGULAR_VELOCITY + 2] = Vector3.Dot(spin, body.AxisZ) * WormContract.ANGULAR_VELOCITY_SCALE;

            for (int jointIndex = 0; jointIndex < WormContract.ACTION_SIZE; jointIndex++)
            {
                obs[WormContract.OBS_JOINT_POSITION + jointIndex] = jointPositions[jointIndex] / WormContract.JOINT_RANGE_RAD;
                obs[WormContract.OBS_JOINT_VELOCITY + jointIndex] = jointVelocities[jointIndex] * WormContract.JOINT_VELOCITY_SCALE;
                obs[WormContract.OBS_PREVIOUS_ACTION + jointIndex] = previousAction[jointIndex];
            }

            // Goal: rotated into B, then only B's horizontal (x, y) part kept and normalised.
            // For a worm lying flat this equals the yaw-only heading error as a unit vector.
            float goalX = Vector3.Dot(body.Goal, body.AxisX);
            float goalY = Vector3.Dot(body.Goal, body.AxisY);
            float norm = Mathf.Sqrt(goalX * goalX + goalY * goalY);
            if (norm > MIN_GOAL_NORM)
            {
                obs[WormContract.OBS_GOAL] = goalX / norm;
                obs[WormContract.OBS_GOAL + 1] = goalY / norm;
            }
            else
            {
                // Pointing straight up or down: no horizontal heading exists. Say "ahead".
                obs[WormContract.OBS_GOAL] = 1f;
                obs[WormContract.OBS_GOAL + 1] = 0f;
            }
        }
    }
}
