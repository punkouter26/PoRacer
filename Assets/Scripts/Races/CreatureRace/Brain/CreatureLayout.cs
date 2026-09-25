using System;
using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// One creature's racing contract, resolved once per play session from its rig and its
    /// observation definition: which body the observation is built in, which body's front
    /// crosses the finish line, the rest pose, the action scale and the observation shape.
    /// Shared by every racer of that creature, whatever simulator steps it.
    /// </summary>
    public sealed class CreatureLayout
    {
        private readonly float[] _restPose;

        private CreatureLayout(CreatureRig rig, CreatureObservationSpec spec, int referenceBody, int leadBody,
                               float actionScale, float targetSpeed)
        {
            Rig = rig;
            ReferenceBody = referenceBody;
            LeadBody = leadBody;
            LeadOffset = new Vector3(rig.ForwardExtent(leadBody), 0f, 0f);
            ActionSize = rig.ActionSize;
            ActionScale = actionScale;
            _restPose = new float[rig.ActionSize];
            for (int actionIndex = 0; actionIndex < _restPose.Length; actionIndex++)
            {
                _restPose[actionIndex] = rig.RestPose[actionIndex];
            }
            LinearVelocityScale = spec.LinearVelocityScale;
            AngularVelocityScale = spec.AngularVelocityScale;
            JointVelocityScale = spec.JointVelocityScale;
            JointPositionsRelativeToRest = spec.JointPositionsRelativeToRest;
            IncludeTargetSpeed = spec.IncludeTargetSpeed;
            TargetSpeed = targetSpeed;
            TargetSpeedObservation = spec.IncludeTargetSpeed && spec.TargetSpeedDivisor != 0f
                ? targetSpeed / spec.TargetSpeedDivisor
                : 0f;
            ObservationSize = spec.Size(rig.ActionSize);
        }

        public CreatureRig Rig { get; }
        public int ReferenceBody { get; }
        public int LeadBody { get; }
        /// <summary>The finish point on the lead body, MuJoCo body frame (its forward reach).</summary>
        public Vector3 LeadOffset { get; }
        public int ActionSize { get; }
        public int ObservationSize { get; }
        public float ActionScale { get; }
        public float LinearVelocityScale { get; }
        public float AngularVelocityScale { get; }
        public float JointVelocityScale { get; }
        public bool JointPositionsRelativeToRest { get; }
        public bool IncludeTargetSpeed { get; }
        public float TargetSpeed { get; }
        public float TargetSpeedObservation { get; }

        public float Rest(int actionIndex)
        {
            return _restPose[actionIndex];
        }

        public void CopyRestPose(float[] into)
        {
            Array.Copy(_restPose, into, _restPose.Length);
        }

        /// <param name="actionScaleOverride">
        /// 0 = the rig's scale. The worm passes its contract's float constant so its numbers stay
        /// bit-identical to the pre-template build.
        /// </param>
        public static bool TryCreate(CreatureRig rig, CreatureObservationSpec spec, string leadBody,
                                     float actionScaleOverride, out CreatureLayout layout, out string error)
        {
            layout = null;
            if (rig == null || spec == null)
            {
                error = "no rig or no observation definition";
                return false;
            }
            int reference = string.IsNullOrEmpty(spec.ReferenceBody) ? rig.TorsoBody : rig.FindBody(spec.ReferenceBody);
            if (reference < 0)
            {
                error = $"the observation's reference body '{spec.ReferenceBody}' is not in the rig";
                return false;
            }
            int lead = string.IsNullOrEmpty(leadBody) ? rig.TorsoBody : rig.FindBody(leadBody);
            if (lead < 0)
            {
                error = $"the lead body '{leadBody}' is not in the rig";
                return false;
            }
            float targetSpeed = spec.TargetSpeed > 0f ? spec.TargetSpeed : rig.TargetSpeed;
            if (spec.IncludeTargetSpeed && !(targetSpeed > 0f))
            {
                error = "the observation includes a target speed, but neither the settings nor the rig give one";
                return false;
            }
            float scale = actionScaleOverride > 0f ? actionScaleOverride : rig.ActionScale;
            layout = new CreatureLayout(rig, spec, reference, lead, scale, float.IsFinite(targetSpeed) ? targetSpeed : 0f);
            error = string.Empty;
            return true;
        }
    }
}
