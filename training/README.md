# training/

Python-side training and export sources for the Isaac Lab / MuJoCo robot ports.
These lived under `Assets/` until 2026-08-29, where Unity imported every `.py`,
`.pt` and `__pycache__` entry into the asset database for no benefit.

| Folder | What it is |
|---|---|
| `biped_sentis/` | MuJoCo + Sentis biped. Produces `Assets/unity_export/MujocoBiped/MujocoBiped.onnx`. |
| `h1/` | Unitree H1 Isaac Lab task. Produces `Assets/unity_export/IsaacH1/IsaacH1.onnx`. |
| `export_tools/` | Per-rig ONNX checks, rig audits and mesh extraction helpers. |

The exported `.onnx` policies and their rigs are the only artifacts that belong
in `Assets/` — nothing in C# references anything in this folder.
