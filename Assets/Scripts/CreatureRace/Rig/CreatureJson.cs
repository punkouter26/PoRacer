using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// A small, strict JSON reader for rig files. JsonUtility cannot read what the trainers
    /// write (null parents, nested arrays such as fromto pairs, name-keyed dictionaries like
    /// restPose, optional keys that must be told apart from zero), and the project only has
    /// Newtonsoft as a transitive package, so the rig parser reads through this instead.
    ///
    /// Values come back as Dictionary&lt;string, object&gt;, List&lt;object&gt;, double, string,
    /// bool or null. Parsed once per play session, never in a physics step.
    /// </summary>
    internal static class CreatureJson
    {
        public static bool TryParse(string text, out object value, out string error)
        {
            value = null;
            if (string.IsNullOrEmpty(text))
            {
                error = "the text is empty";
                return false;
            }
            var reader = new Reader(text);
            try
            {
                reader.SkipWhitespace();
                value = reader.ReadValue();
                reader.SkipWhitespace();
                if (!reader.AtEnd)
                {
                    throw reader.Fail("unexpected text after the top-level value");
                }
            }
            catch (FormatException exception)
            {
                value = null;
                error = exception.Message;
                return false;
            }
            error = string.Empty;
            return true;
        }

        private sealed class Reader
        {
            private readonly string _text;
            private readonly StringBuilder _buffer = new();
            private int _position;

            public Reader(string text)
            {
                _text = text;
            }

            public bool AtEnd => _position >= _text.Length;

            public FormatException Fail(string reason)
            {
                return new FormatException($"JSON error at character {_position}: {reason}");
            }

            public void SkipWhitespace()
            {
                while (!AtEnd && char.IsWhiteSpace(_text[_position]))
                {
                    _position++;
                }
            }

            public object ReadValue()
            {
                if (AtEnd)
                {
                    throw Fail("unexpected end of text");
                }
                char next = _text[_position];
                switch (next)
                {
                    case '{':
                        return ReadObject();
                    case '[':
                        return ReadArray();
                    case '"':
                        return ReadString();
                    case 't':
                        ExpectWord("true");
                        return true;
                    case 'f':
                        ExpectWord("false");
                        return false;
                    case 'n':
                        ExpectWord("null");
                        return null;
                    default:
                        if (next == '-' || char.IsDigit(next))
                        {
                            return ReadNumber();
                        }
                        // Python's json module writes these for non-finite floats.
                        if (next == 'N')
                        {
                            ExpectWord("NaN");
                            return double.NaN;
                        }
                        if (next == 'I')
                        {
                            ExpectWord("Infinity");
                            return double.PositiveInfinity;
                        }
                        throw Fail($"unexpected character '{next}'");
                }
            }

            private Dictionary<string, object> ReadObject()
            {
                var result = new Dictionary<string, object>(StringComparer.Ordinal);
                _position++;
                SkipWhitespace();
                if (!AtEnd && _text[_position] == '}')
                {
                    _position++;
                    return result;
                }
                while (true)
                {
                    SkipWhitespace();
                    if (AtEnd || _text[_position] != '"')
                    {
                        throw Fail("expected a key string");
                    }
                    string key = ReadString();
                    SkipWhitespace();
                    Expect(':');
                    SkipWhitespace();
                    result[key] = ReadValue();
                    SkipWhitespace();
                    if (AtEnd)
                    {
                        throw Fail("unterminated object");
                    }
                    char separator = _text[_position++];
                    if (separator == '}')
                    {
                        return result;
                    }
                    if (separator != ',')
                    {
                        throw Fail("expected ',' or '}'");
                    }
                }
            }

            private List<object> ReadArray()
            {
                var result = new List<object>();
                _position++;
                SkipWhitespace();
                if (!AtEnd && _text[_position] == ']')
                {
                    _position++;
                    return result;
                }
                while (true)
                {
                    SkipWhitespace();
                    result.Add(ReadValue());
                    SkipWhitespace();
                    if (AtEnd)
                    {
                        throw Fail("unterminated array");
                    }
                    char separator = _text[_position++];
                    if (separator == ']')
                    {
                        return result;
                    }
                    if (separator != ',')
                    {
                        throw Fail("expected ',' or ']'");
                    }
                }
            }

            private string ReadString()
            {
                _position++;
                _buffer.Clear();
                while (true)
                {
                    if (AtEnd)
                    {
                        throw Fail("unterminated string");
                    }
                    char next = _text[_position++];
                    if (next == '"')
                    {
                        return _buffer.ToString();
                    }
                    if (next != '\\')
                    {
                        _buffer.Append(next);
                        continue;
                    }
                    if (AtEnd)
                    {
                        throw Fail("unterminated escape");
                    }
                    char escaped = _text[_position++];
                    switch (escaped)
                    {
                        case '"':
                        case '\\':
                        case '/':
                            _buffer.Append(escaped);
                            break;
                        case 'b':
                            _buffer.Append('\b');
                            break;
                        case 'f':
                            _buffer.Append('\f');
                            break;
                        case 'n':
                            _buffer.Append('\n');
                            break;
                        case 'r':
                            _buffer.Append('\r');
                            break;
                        case 't':
                            _buffer.Append('\t');
                            break;
                        case 'u':
                            if (_position + 4 > _text.Length)
                            {
                                throw Fail("short unicode escape");
                            }
                            _buffer.Append((char)int.Parse(_text.Substring(_position, 4), NumberStyles.HexNumber,
                                                           CultureInfo.InvariantCulture));
                            _position += 4;
                            break;
                        default:
                            throw Fail($"unknown escape '\\{escaped}'");
                    }
                }
            }

            private double ReadNumber()
            {
                int start = _position;
                if (_text[_position] == '-')
                {
                    _position++;
                    if (!AtEnd && _text[_position] == 'I')
                    {
                        ExpectWord("Infinity");
                        return double.NegativeInfinity;
                    }
                }
                while (!AtEnd)
                {
                    char next = _text[_position];
                    if (char.IsDigit(next) || next == '.' || next == 'e' || next == 'E' || next == '+' || next == '-')
                    {
                        _position++;
                        continue;
                    }
                    break;
                }
                string token = _text.Substring(start, _position - start);
                if (!double.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out double number))
                {
                    throw Fail($"bad number '{token}'");
                }
                return number;
            }

            private void Expect(char wanted)
            {
                if (AtEnd || _text[_position] != wanted)
                {
                    throw Fail($"expected '{wanted}'");
                }
                _position++;
            }

            private void ExpectWord(string word)
            {
                if (string.CompareOrdinal(_text, _position, word, 0, word.Length) != 0)
                {
                    throw Fail($"expected '{word}'");
                }
                _position += word.Length;
            }
        }
    }
}
