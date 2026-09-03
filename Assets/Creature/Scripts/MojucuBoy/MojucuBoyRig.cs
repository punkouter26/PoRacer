using System;
using System.Collections.Generic;
using System.Globalization;

namespace Creature.MojucuBoy
{
    /// <summary>
    /// Reader for mojucuboy_rig.json, the rig contract emitted by
    /// training/mojucuboy/build_mjcf.py: the actuator order, each joint's range, and the
    /// standing stance.
    ///
    /// Hand-rolled rather than JsonUtility because the file is a nested structure
    /// with a bare array of objects, which JsonUtility cannot deserialise without a
    /// wrapper type per level -- and a silently zero-filled stance here would put
    /// the racer in a splayed pose that looks like a physics bug rather than a
    /// parsing one.
    /// </summary>
    public sealed class MojucuBoyRig
    {
        public string RootBody { get; private set; } = "hips";
        public string[] ActuatorOrder { get; private set; }
        public float[] Stance { get; private set; }
        public float[] RangeLo { get; private set; }
        public float[] RangeHi { get; private set; }

        public static MojucuBoyRig Parse(string json)
        {
            var rig = new MojucuBoyRig
            {
                ActuatorOrder = StringArray(json, "actuator_order"),
            };

            var stance = new List<float>();
            var lo = new List<float>();
            var hi = new List<float>();
            foreach (string entry in Objects(json, "joints"))
            {
                stance.Add(Scalar(entry, "stance_rad"));
                float[] range = FloatArray(entry, "range_rad");
                lo.Add(range[0]);
                hi.Add(range[1]);
            }
            rig.Stance = stance.ToArray();
            rig.RangeLo = lo.ToArray();
            rig.RangeHi = hi.ToArray();

            if (rig.Stance.Length != rig.ActuatorOrder.Length)
            {
                throw new FormatException(
                    $"mojucuboy_rig.json: {rig.ActuatorOrder.Length} actuators but "
                  + $"{rig.Stance.Length} joint entries");
            }
            return rig;
        }

        private static int KeyAt(string json, string key, int from = 0)
        {
            int at = json.IndexOf("\"" + key + "\"", from, StringComparison.Ordinal);
            if (at < 0)
            {
                throw new FormatException($"mojucuboy_rig.json: key '{key}' missing");
            }
            return json.IndexOf(':', at) + 1;
        }

        private static float Scalar(string json, string key)
        {
            int start = KeyAt(json, key);
            int end = json.IndexOfAny(new[] { ',', '}', '\n' }, start);
            return float.Parse(json.Substring(start, end - start).Trim(),
                               CultureInfo.InvariantCulture);
        }

        private static string[] Slice(string json, string key)
        {
            int start = json.IndexOf('[', KeyAt(json, key));
            int end = json.IndexOf(']', start);
            string inner = json.Substring(start + 1, end - start - 1).Trim();
            if (inner.Length == 0)
            {
                return Array.Empty<string>();
            }
            string[] parts = inner.Split(',');
            for (int i = 0; i < parts.Length; i++)
            {
                parts[i] = parts[i].Trim();
            }
            return parts;
        }

        private static string[] StringArray(string json, string key)
        {
            string[] parts = Slice(json, key);
            for (int i = 0; i < parts.Length; i++)
            {
                parts[i] = parts[i].Trim('"');
            }
            return parts;
        }

        private static float[] FloatArray(string json, string key)
        {
            string[] parts = Slice(json, key);
            var values = new float[parts.Length];
            for (int i = 0; i < parts.Length; i++)
            {
                values[i] = float.Parse(parts[i], CultureInfo.InvariantCulture);
            }
            return values;
        }

        /// <summary>Split an array-of-objects value into its top-level object texts.</summary>
        private static List<string> Objects(string json, string key)
        {
            int start = json.IndexOf('[', KeyAt(json, key));
            var results = new List<string>();
            int depth = 0, objectStart = -1;
            for (int i = start; i < json.Length; i++)
            {
                char c = json[i];
                if (c == '{')
                {
                    if (depth == 0)
                    {
                        objectStart = i;
                    }
                    depth++;
                }
                else if (c == '}')
                {
                    depth--;
                    if (depth == 0)
                    {
                        results.Add(json.Substring(objectStart, i - objectStart + 1));
                    }
                }
                else if (c == ']' && depth == 0)
                {
                    break;
                }
            }
            return results;
        }
    }
}
