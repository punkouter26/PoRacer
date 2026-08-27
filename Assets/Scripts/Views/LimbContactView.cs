using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Collision relay attached at runtime to every solid limb collider of a
    /// creature. Unity only delivers collision messages to the GameObject that owns
    /// the collider, so the racer root cannot hear its own footfalls without one of
    /// these per limb. Reports ground contacts up to the racer's CreatureAudioView,
    /// which owns the rate limiting and the actual playback.
    ///
    /// Contacts against anything carrying a body — other limbs, other racers — are
    /// dropped here, so self-collision inside an articulated creature never reaches
    /// the audio path.
    ///
    /// The same contact also drives the impact sparks. Those keep their own gate
    /// rather than borrowing the audio one: a spark is worth showing at a lower
    /// impact than is worth hearing, and the visual budget is spent on the racers
    /// near the camera rather than on the ones that happen to hold an audio voice.
    /// </summary>
    [DisallowMultipleComponent]
    internal sealed class LimbContactView : MonoBehaviour
    {
        // Sparks want a harder landing than a thud does; below this it is a scuff.
        private const float MIN_SPARK_SPEED = 2.6f;
        private const float FULL_SPARK_SPEED = 7f;
        private const float MIN_SPARK_INTERVAL = 0.12f;
        // Past this the sparks are a few pixels and not worth the emit call.
        private const float SPARK_VISIBLE_RANGE = 45f;
        private const float CAMERA_SAMPLE_INTERVAL = 0.25f;

        // Camera.main runs a tagged object lookup, and this class sits in the
        // collision path of every limb of every racer. The position is cached and
        // refreshed on a timer instead, shared by every relay in the scene.
        private static Vector3 _cameraPosition;
        private static float _nextCameraSampleTime;
        private static Transform _cameraTransform;

        private CreatureAudioView _owner;
        private float _nextSparkTime;

        internal void Bind(CreatureAudioView owner)
        {
            _owner = owner;
        }

        private void OnCollisionEnter(Collision collision)
        {
            if (_owner == null || collision.contactCount == 0)
            {
                return;
            }
            // Ground and scenery are static: no rigidbody, no articulation body.
            // Anything else is a limb (ours or a rival's) and stays silent.
            Collider other = collision.collider;
            if (other == null || other.attachedRigidbody != null || other.attachedArticulationBody != null)
            {
                return;
            }
            // GetContact avoids the array allocation that collision.contacts makes.
            ContactPoint contact = collision.GetContact(0);
            float impactSpeed = collision.relativeVelocity.magnitude;
            _owner.ReportLimbImpact(impactSpeed, contact.point);
            TrySparks(impactSpeed, contact);
        }

        private void TrySparks(float impactSpeed, ContactPoint contact)
        {
            if (impactSpeed < MIN_SPARK_SPEED || Time.time < _nextSparkTime)
            {
                return;
            }
            if (!TryGetCameraPosition(out Vector3 cameraPosition))
            {
                return;
            }
            // Squared compare: a distance check per contact is not worth a sqrt.
            float rangeSqr = SPARK_VISIBLE_RANGE * SPARK_VISIBLE_RANGE;
            if ((contact.point - cameraPosition).sqrMagnitude > rangeSqr)
            {
                return;
            }
            _nextSparkTime = Time.time + MIN_SPARK_INTERVAL;
            float strength = Mathf.InverseLerp(MIN_SPARK_SPEED, FULL_SPARK_SPEED, impactSpeed);
            FxUtil.ImpactSparks(contact.point, contact.normal, strength);
        }

        /// <summary>
        /// Camera position for the range test, re-read a few times a second. The
        /// camera moves smoothly and the test has 45 m of slack, so a stale sample
        /// can only ever mis-judge a racer sitting exactly on the boundary.
        /// </summary>
        private static bool TryGetCameraPosition(out Vector3 position)
        {
            if (Time.time >= _nextCameraSampleTime || _cameraTransform == null)
            {
                _nextCameraSampleTime = Time.time + CAMERA_SAMPLE_INTERVAL;
                Camera camera = Camera.main;
                _cameraTransform = camera != null ? camera.transform : null;
            }
            if (_cameraTransform == null)
            {
                position = default;
                return false;
            }
            _cameraPosition = _cameraTransform.position;
            position = _cameraPosition;
            return true;
        }
    }
}
