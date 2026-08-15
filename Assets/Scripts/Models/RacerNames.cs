using System.Collections.Generic;

namespace PoRacer.Models
{
    /// <summary>
    /// Fun racer name pool. Spawn shuffles an index list per race so names are
    /// unique until the pool runs out, then repeats get a numeric suffix.
    /// </summary>
    public static class RacerNames
    {
        private static readonly string[] Pool =
        {
            "Wiggles", "Ziggy", "Turbo", "Noodle", "Sprocket", "Biscuit",
            "Waffles", "Pickle", "Gizmo", "Zoomer", "Doodle", "Squirt",
            "Bubbles", "Chomper", "Dash", "Fidget", "Gadget", "Hopper",
            "Inky", "Jitter", "Klaus", "Loopy", "Munch", "Nibbles",
            "Otto", "Pretzel", "Quibble", "Rocket", "Scooter", "Tango",
            "Umbra", "Vroom", "Wobble", "Xeno", "Yoyo", "Zippy",
            "Blinky", "Crumbs", "Dozer", "Echo", "Flapjack", "Gumbo",
            "Hazel", "Igor", "Jelly", "Kiwi", "Lemmy", "Momo"
        };

        /// <summary>Fills 'order' with a shuffled permutation of pool indices.</summary>
        public static void Shuffle(System.Random rng, List<int> order)
        {
            order.Clear();
            for (int nameIndex = 0; nameIndex < Pool.Length; nameIndex++)
            {
                order.Add(nameIndex);
            }
            for (int swapIndex = order.Count - 1; swapIndex > 0; swapIndex--)
            {
                int otherIndex = rng.Next(swapIndex + 1);
                (order[swapIndex], order[otherIndex]) = (order[otherIndex], order[swapIndex]);
            }
        }

        /// <summary>Name for the nth racer given a shuffled order; repeats get " 2", " 3", ...</summary>
        public static string Get(List<int> order, int racerIndex)
        {
            string name = Pool[order[racerIndex % order.Count]];
            int cycle = racerIndex / order.Count;
            return cycle == 0 ? name : $"{name} {cycle + 1}";
        }
    }
}
