using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using UnityEngine;

namespace IsaacH1
{
    /// <summary>
    /// A small, complete JSON reader plus typed accessors.
    ///
    /// Unity's JsonUtility cannot do what these files need: it refuses a top-level array
    /// (isaac_reference.json's raw form), it cannot represent a nullable nested object
    /// (a rig body has no "joint" when it is the articulation root), and it cannot read
    /// jagged float arrays. It lives in the runtime assembly only so the PlayMode test
    /// assembly can use it without being Editor-only (an Editor-only test assembly is
    /// classified as EditMode, where FixedUpdate never runs). The agent itself never
    /// parses JSON - it reads the generated IsaacH1RigAsset.
    ///
    /// Parses to Dictionary&lt;string, object&gt; / List&lt;object&gt; / string / double /
    /// bool / null.
    /// </summary>
    public static class MiniJson
    {
        public static object Parse(string json)
        {
            int i = 0;
            object v = ParseValue(json, ref i);
            SkipWhite(json, ref i);
            if (i != json.Length)
                throw new FormatException($"trailing content at index {i}");
            return v;
        }

        // ------------------------------------------------------------- accessors --
        public static Dictionary<string, object> Obj(Dictionary<string, object> d, string key)
            => d != null && d.TryGetValue(key, out var v) ? v as Dictionary<string, object> : null;

        public static List<object> Arr(Dictionary<string, object> d, string key)
            => d != null && d.TryGetValue(key, out var v) ? v as List<object> : null;

        public static string Str(Dictionary<string, object> d, string key)
            => d != null && d.TryGetValue(key, out var v) ? v as string : null;

        public static float Num(Dictionary<string, object> d, string key)
            => d != null && d.TryGetValue(key, out var v) && v is double x ? (float)x : 0f;

        public static bool Bool(Dictionary<string, object> d, string key)
            => d != null && d.TryGetValue(key, out var v) && v is bool b && b;

        public static Vector3 Vec3(Dictionary<string, object> d, string key)
        {
            var a = Arr(d, key);
            if (a == null || a.Count < 3) return Vector3.zero;
            return new Vector3(F(a[0]), F(a[1]), F(a[2]));
        }

        /// <summary>Reads 4 numbers in file order. Component meaning is the caller's business.</summary>
        public static Vector4 Vec4(Dictionary<string, object> d, string key)
        {
            var a = Arr(d, key);
            if (a == null || a.Count < 4) return Vector4.zero;
            return new Vector4(F(a[0]), F(a[1]), F(a[2]), F(a[3]));
        }

        public static string[] StrArray(Dictionary<string, object> d, string key)
        {
            var a = Arr(d, key);
            if (a == null) return Array.Empty<string>();
            var outp = new string[a.Count];
            for (int i = 0; i < a.Count; i++) outp[i] = a[i] as string;
            return outp;
        }

        public static float[] FloatArray(Dictionary<string, object> d, string key)
        {
            var a = Arr(d, key);
            if (a == null) return Array.Empty<float>();
            var outp = new float[a.Count];
            for (int i = 0; i < a.Count; i++) outp[i] = F(a[i]);
            return outp;
        }

        static float F(object o) => o is double d ? (float)d : 0f;

        // ---------------------------------------------------------------- parser --
        static object ParseValue(string s, ref int i)
        {
            SkipWhite(s, ref i);
            if (i >= s.Length) throw new FormatException("unexpected end of input");
            char c = s[i];
            switch (c)
            {
                case '{': return ParseObject(s, ref i);
                case '[': return ParseArray(s, ref i);
                case '"': return ParseString(s, ref i);
                case 't': Expect(s, ref i, "true"); return true;
                case 'f': Expect(s, ref i, "false"); return false;
                case 'n': Expect(s, ref i, "null"); return null;
                default: return ParseNumber(s, ref i);
            }
        }

        static Dictionary<string, object> ParseObject(string s, ref int i)
        {
            var d = new Dictionary<string, object>();
            i++; // '{'
            SkipWhite(s, ref i);
            if (i < s.Length && s[i] == '}') { i++; return d; }
            while (true)
            {
                SkipWhite(s, ref i);
                string key = ParseString(s, ref i);
                SkipWhite(s, ref i);
                if (i >= s.Length || s[i] != ':') throw new FormatException($"expected ':' at {i}");
                i++;
                d[key] = ParseValue(s, ref i);
                SkipWhite(s, ref i);
                if (i >= s.Length) throw new FormatException("unterminated object");
                if (s[i] == ',') { i++; continue; }
                if (s[i] == '}') { i++; return d; }
                throw new FormatException($"expected ',' or '}}' at {i}");
            }
        }

        static List<object> ParseArray(string s, ref int i)
        {
            var l = new List<object>();
            i++; // '['
            SkipWhite(s, ref i);
            if (i < s.Length && s[i] == ']') { i++; return l; }
            while (true)
            {
                l.Add(ParseValue(s, ref i));
                SkipWhite(s, ref i);
                if (i >= s.Length) throw new FormatException("unterminated array");
                if (s[i] == ',') { i++; continue; }
                if (s[i] == ']') { i++; return l; }
                throw new FormatException($"expected ',' or ']' at {i}");
            }
        }

        static string ParseString(string s, ref int i)
        {
            if (s[i] != '"') throw new FormatException($"expected '\"' at {i}");
            i++;
            var sb = new StringBuilder();
            while (i < s.Length)
            {
                char c = s[i++];
                if (c == '"') return sb.ToString();
                if (c != '\\') { sb.Append(c); continue; }
                if (i >= s.Length) break;
                char e = s[i++];
                switch (e)
                {
                    case '"': sb.Append('"'); break;
                    case '\\': sb.Append('\\'); break;
                    case '/': sb.Append('/'); break;
                    case 'b': sb.Append('\b'); break;
                    case 'f': sb.Append('\f'); break;
                    case 'n': sb.Append('\n'); break;
                    case 'r': sb.Append('\r'); break;
                    case 't': sb.Append('\t'); break;
                    case 'u':
                        sb.Append((char)Convert.ToInt32(s.Substring(i, 4), 16));
                        i += 4;
                        break;
                    default: throw new FormatException($"bad escape '\\{e}' at {i}");
                }
            }
            throw new FormatException("unterminated string");
        }

        static object ParseNumber(string s, ref int i)
        {
            int start = i;
            if (i < s.Length && (s[i] == '-' || s[i] == '+')) i++;
            while (i < s.Length && (char.IsDigit(s[i]) || s[i] == '.' ||
                                    s[i] == 'e' || s[i] == 'E' || s[i] == '-' || s[i] == '+')) i++;
            string tok = s.Substring(start, i - start);
            if (!double.TryParse(tok, NumberStyles.Float, CultureInfo.InvariantCulture, out double v))
                throw new FormatException($"bad number '{tok}' at {start}");
            return v;
        }

        static void Expect(string s, ref int i, string lit)
        {
            if (i + lit.Length > s.Length || string.CompareOrdinal(s, i, lit, 0, lit.Length) != 0)
                throw new FormatException($"expected '{lit}' at {i}");
            i += lit.Length;
        }

        static void SkipWhite(string s, ref int i)
        {
            while (i < s.Length && char.IsWhiteSpace(s[i])) i++;
        }
    }
}
