using System.Collections.Generic;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Hazard trigger volume built by Systems_TrackBuilder: any ArticulationBody
    /// part inside gets a viscous drag force, so creatures wade slowly through
    /// mud. Bodies are collected in OnTriggerStay and forced once per
    /// FixedUpdate so multi-collider limbs are never double-damped.
    /// </summary>
    [RequireComponent(typeof(BoxCollider))]
    public sealed class MudZoneView : MonoBehaviour
    {
        // Deceleration per m/s of speed; creatures crawl at ~0.5-2 m/s.
        private const float MUD_DRAG_PER_SPEED = 4f;

        private readonly List<ArticulationBody> _bodiesInMud = new();

        private void OnTriggerStay(Collider other)
        {
            ArticulationBody body = other.attachedArticulationBody;
            if (body != null && !_bodiesInMud.Contains(body))
            {
                _bodiesInMud.Add(body);
            }
        }

        private void FixedUpdate()
        {
            for (int bodyIndex = 0; bodyIndex < _bodiesInMud.Count; bodyIndex++)
            {
                ArticulationBody body = _bodiesInMud[bodyIndex];
                if (body != null)
                {
                    body.AddForce(-body.linearVelocity * (MUD_DRAG_PER_SPEED * body.mass));
                }
            }
            _bodiesInMud.Clear();
        }
    }
}
