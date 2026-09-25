using System;
using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// The one place an observation is assembled, for every creature and every simulator
    /// (layout: <see cref="CreatureObservationSpec"/>). Each view converts its physics state
    /// into a <see cref="CreatureBodyState"/>; from there on every racer goes through
    /// identical code, so a difference in how they race cannot come from what they see.
    /// </summary>
    public static class CreatureObservation
    {
        private const int GRAVITY = 0;
        private const int LINEAR_VELOCITY = 3;
        private const int ANGULAR_VELOCITY = 6;
        private const int JOINT_POSITION = 9;
        private const float MIN_GOAL_NORM = 1e-6f;

        public static void Build(CreatureLayout layout, in CreatureBodyState body, float[] jointPositions,
                                 float[] jointVelocities, float[] previousAction, float[] obs)
        {
            if (obs == null || obs.Length != layout.ObservationSize)
            {
                throw new ArgumentException($"obs must be length {layout.ObservationSize}", nameof(obs));
            }
            int actions = layout.ActionSize;
            int jointVelocity = JOINT_POSITION + actions;
            int previous = jointVelocity + actions;
            int goal = previous + actions;

            // Gravity (0, 0, -1) in B: R^T g, i.e. minus the z component of each B axis.
            obs[GRAVITY] = -body.AxisX.z;
            obs[GRAVITY + 1] = -body.AxisY.z;
            obs[GRAVITY + 2] = -body.AxisZ.z;

            Vector3 velocity = body.LinearVelocity;
            obs[LINEAR_VELOCITY] = Vector3.Dot(velocity, body.AxisX) * layout.LinearVelocityScale;
            obs[LINEAR_VELOCITY + 1] = Vector3.Dot(velocity, body.AxisY) * layout.LinearVelocityScale;
            obs[LINEAR_VELOCITY + 2] = Vector3.Dot(velocity, body.AxisZ) * layout.LinearVelocityScale;

            Vector3 spin = body.AngularVelocity;
            obs[ANGULAR_VELOCITY] = Vector3.Dot(spin, body.AxisX) * layout.AngularVelocityScale;
            obs[ANGULAR_VELOCITY + 1] = Vector3.Dot(spin, body.AxisY) * layout.AngularVelocityScale;
            obs[ANGULAR_VELOCITY + 2] = Vector3.Dot(spin, body.AxisZ) * layout.AngularVelocityScale;

            for (int jointIndex = 0; jointIndex < actions; jointIndex++)
            {
                float position = layout.JointPositionsRelativeToRest
                    ? jointPositions[jointIndex] - layout.Rest(jointIndex)
                    : jointPositions[jointIndex];
                obs[JOINT_POSITION + jointIndex] = position / layout.ActionScale;
                obs[jointVelocity + jointIndex] = jointVelocities[jointIndex] * layout.JointVelocityScale;
                obs[previous + jointIndex] = previousAction[jointIndex];
            }

            // Goal: rotated into B, then only B's horizontal (x, y) part kept and normalised.
            float goalX = Vector3.Dot(body.Goal, body.AxisX);
            float goalY = Vector3.Dot(body.Goal, body.AxisY);
            float norm = Mathf.Sqrt(goalX * goalX + goalY * goalY);
            if (norm > MIN_GOAL_NORM)
            {
                obs[goal] = goalX / norm;
                obs[goal + 1] = goalY / norm;
            }
            else
            {
                // Pointing straight up or down: no horizontal heading exists. Say "ahead".
                obs[goal] = 1f;
                obs[goal + 1] = 0f;
            }

            if (layout.IncludeTargetSpeed)
            {
                obs[goal + 2] = layout.TargetSpeedObservation;
            }
        }
    }
}
