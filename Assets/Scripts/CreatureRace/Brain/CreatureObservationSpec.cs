using System;
using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// What a creature's policy observes, as trained (WORM_SPEC, QUAD_SPEC). Every creature
    /// shares one layout, built in the reference body's frame (MuJoCo convention):
    ///
    /// <code>
    ///   gravity in B (3)                     x1
    ///   B linear velocity (3)                x linear scale   (0.5)
    ///   B angular velocity (3)               x angular scale  (0.25)
    ///   joint positions (A), action order    / action scale   (optionally minus the rest pose)
    ///   joint velocities (A)                 x joint scale    (0.1)
    ///   previous action (A)                  x1
    ///   goal direction in B, horizontal (2)  unit
    ///   target speed (1, optional)           / divisor        (quad: 1.49 / 2)
    /// </code>
    ///
    /// Size = 9 + 3A + 2 (+1): the worm's 35 (A = 8), the quad's 36 (A = 8, with target speed).
    /// </summary>
    [Serializable]
    public sealed class CreatureObservationSpec
    {
        [Tooltip("Body B the observation is expressed in (worm: seg2, quad: the torso). Empty = the rig's torso.")]
        [SerializeField] private string _referenceBody = string.Empty;
        [SerializeField] private float _linearVelocityScale = 0.5f;
        [SerializeField] private float _angularVelocityScale = 0.25f;
        [SerializeField] private float _jointVelocityScale = 0.1f;
        [Tooltip("Observe (q - rest) / scale instead of q / scale. Off for both creatures so far (rest = 0).")]
        [SerializeField] private bool _jointPositionsRelativeToRest;
        [Tooltip("Append the task's target speed / divisor as the last value (QUAD_SPEC).")]
        [SerializeField] private bool _includeTargetSpeed;
        [Tooltip("m/s; 0 = the rig's task.targetSpeed.")]
        [SerializeField] private float _targetSpeed;
        [SerializeField] private float _targetSpeedDivisor = 2f;

        public string ReferenceBody => _referenceBody ?? string.Empty;
        public float LinearVelocityScale => _linearVelocityScale;
        public float AngularVelocityScale => _angularVelocityScale;
        public float JointVelocityScale => _jointVelocityScale;
        public bool JointPositionsRelativeToRest => _jointPositionsRelativeToRest;
        public bool IncludeTargetSpeed => _includeTargetSpeed;
        public float TargetSpeed => _targetSpeed;
        public float TargetSpeedDivisor => _targetSpeedDivisor;

        public int Size(int actionSize)
        {
            return 9 + 3 * actionSize + 2 + (_includeTargetSpeed ? 1 : 0);
        }
    }
}
