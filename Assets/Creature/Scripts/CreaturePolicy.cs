using System;
using UnityEngine;

namespace Creature {

/// <summary>One dense layer, kernel stored row-major as [inSize * outSize].</summary>
[Serializable]
public class PolicyLayer {
  public int inSize;
  public int outSize;
  public float[] kernel;
  public float[] bias;
}

/// <summary>Schema of Assets/Creature/policy.json, written by mjx_training/export_policy.py.</summary>
[Serializable]
public class PolicyData {
  public string format;
  public int obsSize;
  public int actionSize;
  public string activation;
  public float ctrlDt;
  public int nSubsteps;
  public float[] mean;
  public float[] std;
  public float[] homeJointPos;
  public PolicyLayer[] layers;
}

/// <summary>
/// Deterministic inference for a Brax PPO policy exported from MJX.
///
/// Mirrors mjx_training/export_policy.py exactly:
///   x = (obs - mean) / std
///   x = silu(x * W_i + b_i)        for each hidden layer
///   out = x * W_last + b_last      (2 * actionSize wide)
///   action = tanh(out[0 .. actionSize])
/// </summary>
public class CreaturePolicy {
  public const string ExpectedFormat = "mjx-unity-policy/1";

  readonly PolicyData _data;
  readonly float[][] _scratch;

  public int ObsSize => _data.obsSize;
  public int ActionSize => _data.actionSize;
  public float CtrlDt => _data.ctrlDt;
  public int NSubsteps => _data.nSubsteps;

  /// <summary>
  /// Joint angles of the MJCF "home" keyframe, in actuator order, or null if the
  /// export predates this field. Unity's MJCF importer ignores &lt;keyframe&gt;
  /// entirely, so the training start pose has to arrive with the policy.
  /// </summary>
  public float[] HomeJointPos =>
      (_data.homeJointPos != null && _data.homeJointPos.Length == _data.actionSize)
          ? _data.homeJointPos : null;

  CreaturePolicy(PolicyData data) {
    _data = data;
    _scratch = new float[data.layers.Length][];
    for (int i = 0; i < data.layers.Length; i++) {
      _scratch[i] = new float[data.layers[i].outSize];
    }
  }

  /// <summary>Parses and validates a policy JSON. Throws on any inconsistency.</summary>
  public static CreaturePolicy FromJson(string json) {
    var data = JsonUtility.FromJson<PolicyData>(json);
    if (data == null) {
      throw new ArgumentException("policy JSON could not be parsed.");
    }
    if (data.format != ExpectedFormat) {
      throw new ArgumentException(
          $"policy format '{data.format}' != expected '{ExpectedFormat}'.");
    }
    if (data.layers == null || data.layers.Length == 0) {
      throw new ArgumentException("policy JSON has no layers.");
    }
    if (data.mean == null || data.mean.Length != data.obsSize ||
        data.std == null || data.std.Length != data.obsSize) {
      throw new ArgumentException(
          $"normalizer length != obsSize ({data.obsSize}).");
    }
    if (data.layers[0].inSize != data.obsSize) {
      throw new ArgumentException(
          $"first layer expects {data.layers[0].inSize} inputs but obsSize is {data.obsSize}.");
    }
    int tail = data.layers[data.layers.Length - 1].outSize;
    if (tail != 2 * data.actionSize) {
      throw new ArgumentException(
          $"final layer emits {tail}, expected {2 * data.actionSize} (mean+std).");
    }
    for (int i = 0; i < data.layers.Length; i++) {
      var l = data.layers[i];
      if (l.kernel == null || l.kernel.Length != l.inSize * l.outSize) {
        throw new ArgumentException($"layer {i} kernel size mismatch.");
      }
      if (l.bias == null || l.bias.Length != l.outSize) {
        throw new ArgumentException($"layer {i} bias size mismatch.");
      }
      if (i > 0 && l.inSize != data.layers[i - 1].outSize) {
        throw new ArgumentException($"layer {i} input does not match layer {i - 1} output.");
      }
    }
    return new CreaturePolicy(data);
  }

  static float Silu(float x) => x / (1f + Mathf.Exp(-x));

  /// <summary>Writes actionSize values in [-1, 1] into <paramref name="action"/>.</summary>
  public void Evaluate(float[] obs, float[] action) {
    if (obs.Length != _data.obsSize) {
      throw new ArgumentException($"expected {_data.obsSize} observations, got {obs.Length}.");
    }
    if (action.Length != _data.actionSize) {
      throw new ArgumentException($"expected {_data.actionSize} action slots, got {action.Length}.");
    }

    // Layer 0 consumes the normalized observation; later layers consume scratch.
    var last = _data.layers.Length - 1;
    for (int li = 0; li <= last; li++) {
      var layer = _data.layers[li];
      var outBuf = _scratch[li];
      var bias = layer.bias;
      var kernel = layer.kernel;
      int outSize = layer.outSize;

      for (int j = 0; j < outSize; j++) {
        outBuf[j] = bias[j];
      }
      for (int i = 0; i < layer.inSize; i++) {
        float xi = li == 0
            ? (obs[i] - _data.mean[i]) / _data.std[i]
            : _scratch[li - 1][i];
        if (xi == 0f) {
          continue;
        }
        int row = i * outSize;
        for (int j = 0; j < outSize; j++) {
          outBuf[j] += xi * kernel[row + j];
        }
      }
      if (li != last) {
        for (int j = 0; j < outSize; j++) {
          outBuf[j] = Silu(outBuf[j]);
        }
      }
    }

    // NormalTanhDistribution.mode() == tanh(mean); the std half is unused.
    var head = _scratch[last];
    for (int a = 0; a < _data.actionSize; a++) {
      action[a] = (float)Math.Tanh(head[a]);
    }
  }
}

}  // namespace Creature
