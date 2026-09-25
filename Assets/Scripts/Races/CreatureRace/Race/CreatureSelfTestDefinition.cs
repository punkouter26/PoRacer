using System;
using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// One self-test of a creature, as data. Every racer of the scene is spawned, held at
    /// rest while it settles, then driven with a fixed action instead of its brain and
    /// measured, each in its own physics; every one must pass (a NO BRAIN racer too).
    ///
    /// RestPose: all actions 0. Passes when the lead point moved less than
    /// <see cref="MaxLeadDisplacement"/>, the speed is below <see cref="MaxSpeed"/>, every joint
    /// stayed within <see cref="MaxJointFromRest"/> of its rest angle, the reference body sits
    /// at <see cref="ExpectedReferenceHeight"/> +/- tolerance, and (when
    /// <see cref="MinUpright"/> is above -1) it is still upright.
    ///
    /// JointSign: <see cref="Joint"/> driven to rest + <see cref="TargetRad"/>. Passes when the
    /// joint reads more than <see cref="MinJointRad"/> above rest AND a probe point on body B,
    /// seen from body A (MuJoCo convention, in A's frame), moved along <see cref="ProbeAxis"/>
    /// by more than <see cref="MinProbeOffset"/> - the sign convention of the trainers.
    /// </summary>
    [Serializable]
    public sealed class CreatureSelfTestDefinition
    {
        [Tooltip("Report and file name, e.g. \"YawSignTest\" (Logs/<prefix>_selftest_<name>_<stamp>.json).")]
        [SerializeField] private string _name = string.Empty;
        [Tooltip("Short name for the CLI, e.g. \"yaw\".")]
        [SerializeField] private string _alias = string.Empty;
        [SerializeField] private CreatureSelfTestKind _kind = CreatureSelfTestKind.RestPose;
        [SerializeField] private float _defaultSeconds = 2f;
        [TextArea(2, 4)]
        [SerializeField] private string _expectation = string.Empty;

        [Header("Joint sign")]
        [Tooltip("Action (actuator or joint) name to drive.")]
        [SerializeField] private string _joint = string.Empty;
        [SerializeField] private float _targetRad = 0.5f;
        [SerializeField] private float _minJointRad = 0.3f;
        [Tooltip("Body A: the frame the probe is measured in.")]
        [SerializeField] private string _probeBodyA = string.Empty;
        [Tooltip("Body B: carries the probe point.")]
        [SerializeField] private string _probeBodyB = string.Empty;
        [Tooltip("The probe point on B, MuJoCo body frame (e.g. a foot).")]
        [SerializeField] private Vector3 _probePoint = Vector3.zero;
        [Tooltip("Expected direction of the probe in A's frame, MuJoCo convention (x forward, y left, z up).")]
        [SerializeField] private Vector3 _probeAxis = Vector3.forward;
        [SerializeField] private float _minProbeOffset = 0.02f;
        [Tooltip("Measure the change since GO instead of the absolute offset.")]
        [SerializeField] private bool _probeRelativeToRelease;
        [Tooltip("Verdict wording, e.g. \"segment1 to the worm's right (Unity +x)\".")]
        [SerializeField] private string _probeLabel = string.Empty;

        [Header("Rest pose")]
        [SerializeField] private float _maxLeadDisplacement = 0.02f;
        [SerializeField] private float _maxSpeed = 0.01f;
        [SerializeField] private float _maxJointFromRest = 0.05f;
        [SerializeField] private float _expectedReferenceHeight;
        [SerializeField] private float _referenceHeightTolerance = 0.01f;
        [Tooltip("Minimum reference up . world up at the end; -1 = not checked.")]
        [SerializeField] private float _minUpright = -1f;

        public string Name => _name ?? string.Empty;
        public string Alias => _alias ?? string.Empty;
        public CreatureSelfTestKind Kind => _kind;
        public float DefaultSeconds => _defaultSeconds;
        public string Expectation => _expectation ?? string.Empty;
        public string Joint => _joint ?? string.Empty;
        public float TargetRad => _targetRad;
        public float MinJointRad => _minJointRad;
        public string ProbeBodyA => _probeBodyA ?? string.Empty;
        public string ProbeBodyB => _probeBodyB ?? string.Empty;
        public Vector3 ProbePoint => _probePoint;
        public Vector3 ProbeAxis => _probeAxis;
        public float MinProbeOffset => _minProbeOffset;
        public bool ProbeRelativeToRelease => _probeRelativeToRelease;
        public string ProbeLabel => _probeLabel ?? string.Empty;
        public float MaxLeadDisplacement => _maxLeadDisplacement;
        public float MaxSpeed => _maxSpeed;
        public float MaxJointFromRest => _maxJointFromRest;
        public float ExpectedReferenceHeight => _expectedReferenceHeight;
        public float ReferenceHeightTolerance => _referenceHeightTolerance;
        public float MinUpright => _minUpright;

        public bool Matches(string key)
        {
            return string.Equals(key, Name, StringComparison.OrdinalIgnoreCase)
                || string.Equals(key, Alias, StringComparison.OrdinalIgnoreCase);
        }
    }
}
