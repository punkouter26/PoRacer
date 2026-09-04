using System.Collections.Generic;
using UnityEngine;

namespace PoRacer.Models
{
    /// <summary>
    /// The produce that rains onto the track when a race ends: every model in
    /// the KIRI fruit-and-veg pack, listed once by Editor_BuildFruitCatalog.
    /// Static data only; Systems_FruitPour picks from it at random.
    /// </summary>
    public sealed class FruitCatalog : ScriptableObject
    {
        [SerializeField] private GameObject[] _models = System.Array.Empty<GameObject>();

        public IReadOnlyList<GameObject> Models => _models;
    }
}
