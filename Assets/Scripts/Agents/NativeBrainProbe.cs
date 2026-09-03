using UnityEngine;

namespace PoRacer.Agents
{
    /// <summary>
    /// Diagnostic tag for a racer whose policy is NOT run by ML-Agents.
    ///
    /// The debug overlay reports the brain roster by walking the scene for
    /// BehaviorParameters, which is how it reads observation and action widths
    /// and whether a model actually bound. Four racers have no such component -
    /// IsaacBox and IsaacH1 run their own Inference Engine worker, MojucuBoy and
    /// Fido run a MuJoCo policy read from JSON - so without this they would be
    /// invisible to the overlay, or worse, read as missing brains.
    ///
    /// Attached at Awake by the adapter that owns the policy, because the adapter
    /// is the only thing that knows where the weights came from.
    /// </summary>
    public sealed class NativeBrainProbe : MonoBehaviour
    {
        [SerializeField] private string _creatureName;
        [SerializeField] private string _policySource;
        [SerializeField] private int _observationCount;
        [SerializeField] private int _actionCount;
        [SerializeField] private bool _hasPolicy;

        public string CreatureName => _creatureName;

        /// <summary>Short, human-readable origin of the weights, e.g. "IE/Burst".</summary>
        public string PolicySource => _policySource;

        public int ObservationCount => _observationCount;

        public int ActionCount => _actionCount;

        /// <summary>False when the weights failed to bind — the case worth seeing.</summary>
        public bool HasPolicy => _hasPolicy;

        /// <summary>
        /// Adds (or refreshes) the probe on <paramref name="host"/>. Safe to call
        /// from Awake on every spawn: a racer that is pooled and respawned keeps
        /// one probe rather than accumulating them.
        /// </summary>
        public static void Attach(GameObject host, string creatureName, string policySource,
            int observationCount, int actionCount, bool hasPolicy)
        {
            if (host == null)
            {
                return;
            }
            if (!host.TryGetComponent(out NativeBrainProbe probe))
            {
                probe = host.AddComponent<NativeBrainProbe>();
            }
            probe._creatureName = creatureName;
            probe._policySource = policySource;
            probe._observationCount = observationCount;
            probe._actionCount = actionCount;
            probe._hasPolicy = hasPolicy;
        }
    }
}
