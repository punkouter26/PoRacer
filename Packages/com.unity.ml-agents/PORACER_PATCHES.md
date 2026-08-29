# Local patches to com.unity.ml-agents 4.1.0

This package is **embedded**, not pulled from the registry, so that the fix below
can exist. It is a verbatim copy of `com.unity.ml-agents@4.1.0` with
`Documentation~/` removed, plus the change listed here.

**If you upgrade ML-Agents, re-apply this patch or delete the embedded copy and
put the version back in `Packages/manifest.json`.**

---

## 1. `Runtime/Inference/TensorProxy.cs` — null guard in the finalizer path

`~TensorProxy()` calls `Dispose()`, which read `data.dataOnBackend.backendType`
with no null check — while the very next line used `data?.Dispose()`, so null was
already expected there.

`ModelRunner.FetchSentisOutputs` allocates a fresh `TensorProxy` per output on
**every decision**, wrapping a worker-owned tensor from `PeekOutput()`, and drops
the previous ones. Once the worker releases that backing store, `data` or
`data.dataOnBackend` is null, and the discarded proxies throw when finalized.

Measured on a Pixel 9 Pro, release build, 10 racers:

    3,553 NullReferenceExceptions in ~2 minutes  ->  ~30 per second
    E Unity : at Unity.MLAgents.Inference.TensorProxy.Dispose ()
    E Unity : at Unity.MLAgents.Inference.TensorProxy.Finalize ()

The guarded body is a no-op for this project regardless — `SentisModelInfo`
constructs its `Worker` with `DeviceType.CPU`, so `backendType != CPU` is never
true. The patch removes pure waste; it changes no behaviour.

```diff
-            if (data.dataOnBackend.backendType != BackendType.CPU)
-            {
-                data?.Dispose();
-            }
+            var backing = data?.dataOnBackend;
+            if (backing != null && backing.backendType != BackendType.CPU)
+            {
+                data.Dispose();
+            }
```

Worth reporting upstream: the bug is in the package, not in how this project uses it.
