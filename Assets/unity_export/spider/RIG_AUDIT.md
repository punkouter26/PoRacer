# Rig audit - spider (Isaac Lab export -> Unity PhysX 4)

kp=25.0 N.m/rad, kd=1.0 N.m.s/rad, effort=15.0 N.m, velocity_limit_sim=12.0 rad/s, Isaac sim_dt=0.008333 s (decimation 4, policy_dt=0.033333 s)

## 1. Light links and small inertia

| link | mass kg | I principal (kg.m2) | neighbours | mass ratio to neighbours |
|---|---|---|---|---|
| body | 2.0 | 8.00e-03, 8.00e-03, 8.00e-03 | L1_femur, L2_femur, L3_femur, L4_femur, R1_femur, R2_femur, R3_femur, R4_femur | L1_femur: 20.00, L2_femur: 20.00, L3_femur: 20.00, L4_femur: 20.00, R1_femur: 20.00, R2_femur: 20.00, R3_femur: 20.00, R4_femur: 20.00 |
| L1_femur | 0.1 | 2.23e-04, 2.23e-04, 2.00e-05 | L1_tibia, body | L1_tibia: 0.83, body: 0.05 |
| L1_tibia | 0.12 | 6.83e-04, 6.83e-04, 1.35e-05 | L1_femur | L1_femur: 1.20 |
| L2_femur | 0.1 | 2.23e-04, 2.23e-04, 2.00e-05 | L2_tibia, body | L2_tibia: 0.83, body: 0.05 |
| L2_tibia | 0.12 | 6.83e-04, 6.83e-04, 1.35e-05 | L2_femur | L2_femur: 1.20 |
| L3_femur | 0.1 | 2.23e-04, 2.23e-04, 2.00e-05 | L3_tibia, body | L3_tibia: 0.83, body: 0.05 |
| L3_tibia | 0.12 | 6.83e-04, 6.83e-04, 1.35e-05 | L3_femur | L3_femur: 1.20 |
| L4_femur | 0.1 | 2.23e-04, 2.23e-04, 2.00e-05 | L4_tibia, body | L4_tibia: 0.83, body: 0.05 |
| L4_tibia | 0.12 | 6.83e-04, 6.83e-04, 1.35e-05 | L4_femur | L4_femur: 1.20 |
| R1_femur | 0.1 | 2.23e-04, 2.23e-04, 2.00e-05 | R1_tibia, body | R1_tibia: 0.83, body: 0.05 |
| R1_tibia | 0.12 | 6.83e-04, 6.83e-04, 1.35e-05 | R1_femur | R1_femur: 1.20 |
| R2_femur | 0.1 | 2.23e-04, 2.23e-04, 2.00e-05 | R2_tibia, body | R2_tibia: 0.83, body: 0.05 |
| R2_tibia | 0.12 | 6.83e-04, 6.83e-04, 1.35e-05 | R2_femur | R2_femur: 1.20 |
| R3_femur | 0.1 | 2.23e-04, 2.23e-04, 2.00e-05 | R3_tibia, body | R3_tibia: 0.83, body: 0.05 |
| R3_tibia | 0.12 | 6.83e-04, 6.83e-04, 1.35e-05 | R3_femur | R3_femur: 1.20 |
| R4_femur | 0.1 | 2.23e-04, 2.23e-04, 2.00e-05 | R4_tibia, body | R4_tibia: 0.83, body: 0.05 |
| R4_tibia | 0.12 | 6.83e-04, 6.83e-04, 1.35e-05 | R4_femur | R4_femur: 1.20 |

Links lighter than 10 % of a neighbour: L1_femur (0.050), L2_femur (0.050), L3_femur (0.050), L4_femur (0.050), R1_femur (0.050), R2_femur (0.050), R3_femur (0.050), R4_femur (0.050) (worst ratio body:femur = 20:1, below the 50:1 retrain threshold).
Links with a principal inertia below 1e-4 kg.m2: L1_femur (2.0e-05), L1_tibia (1.3e-05), L2_femur (2.0e-05), L2_tibia (1.3e-05), L3_femur (2.0e-05), L3_tibia (1.3e-05), L4_femur (2.0e-05), L4_tibia (1.3e-05), R1_femur (2.0e-05), R1_tibia (1.3e-05), R2_femur (2.0e-05), R2_tibia (1.3e-05), R3_femur (2.0e-05), R3_tibia (1.3e-05), R4_femur (2.0e-05), R4_tibia (1.3e-05) - that is the long-axis (roll) component of the cylinders, not the joint-axis component.
Recommendation: keep the URDF masses (massFloor = 0; the 20:1 body:femur ratio is inside PhysX 4 tolerance) and floor each principal inertia at 1e-4 kg.m2 (inertiaFloor = 1e-4 on the Agent). That raises tibia roll inertia 7x and femur roll inertia 5x - a DoF no joint drives - so gait is unaffected while the solver conditioning improves. If the femur ever has to be lightened further (ratio > 50:1) retrain with heavier femurs instead.

## 2. Joint velocities in the Isaac reference (reconstructed from obs[16:32] / 0.1)

| joint | max abs rad/s | p99 abs rad/s |
|---|---|---|
| L1_hip | 13.12 | 12.01 |
| L1_knee | 37.34 | 28.69 |
| L2_hip | 15.26 | 13.90 |
| L2_knee | 30.66 | 24.76 |
| L3_hip | 13.34 | 12.48 |
| L3_knee | 89.79 | 30.09 |
| L4_hip | 14.72 | 13.55 |
| L4_knee | 75.15 | 32.62 |
| R1_hip | 14.75 | 13.52 |
| R1_knee | 21.69 | 19.17 |
| R2_hip | 16.43 | 14.44 |
| R2_knee | 61.71 | 45.60 |
| R3_hip | 16.01 | 13.55 |
| R3_knee | 45.78 | 37.77 |
| R4_hip | 13.60 | 13.34 |
| R4_knee | 14.82 | 12.93 |

Overall max abs joint velocity = 89.79 rad/s vs velocity_limit_sim = 12.0. EXCEEDS the limit -> Isaac did not enforce it; set enforceVelocityLimit = false.

## 3. Explicit-PD stability bound  kd*dt / I_joint  (must stay < 2; < 1 comfortable)

| joint | I about axis (kg.m2, parallel-axis, whole subtree) | project 0.02 | 1/60 | 1/120 (Isaac) | 1/240 | 1/480 | 1/960 |
|---|---|---|---|---|---|---|---|
| L1_hip | 5.381e-03 | 3.72 | 3.10 | 1.55 | 0.77 | 0.39 | 0.19 |
| L1_knee | 2.711e-03 | 7.38 | 6.15 | 3.07 | 1.54 | 0.77 | 0.38 |
| L2_hip | 5.381e-03 | 3.72 | 3.10 | 1.55 | 0.77 | 0.39 | 0.19 |
| L2_knee | 2.711e-03 | 7.38 | 6.15 | 3.07 | 1.54 | 0.77 | 0.38 |
| L3_hip | 5.381e-03 | 3.72 | 3.10 | 1.55 | 0.77 | 0.39 | 0.19 |
| L3_knee | 2.711e-03 | 7.38 | 6.15 | 3.07 | 1.54 | 0.77 | 0.38 |
| L4_hip | 5.381e-03 | 3.72 | 3.10 | 1.55 | 0.77 | 0.39 | 0.19 |
| L4_knee | 2.711e-03 | 7.38 | 6.15 | 3.07 | 1.54 | 0.77 | 0.38 |
| R1_hip | 5.381e-03 | 3.72 | 3.10 | 1.55 | 0.77 | 0.39 | 0.19 |
| R1_knee | 2.711e-03 | 7.38 | 6.15 | 3.07 | 1.54 | 0.77 | 0.38 |
| R2_hip | 5.381e-03 | 3.72 | 3.10 | 1.55 | 0.77 | 0.39 | 0.19 |
| R2_knee | 2.711e-03 | 7.38 | 6.15 | 3.07 | 1.54 | 0.77 | 0.38 |
| R3_hip | 5.381e-03 | 3.72 | 3.10 | 1.55 | 0.77 | 0.39 | 0.19 |
| R3_knee | 2.711e-03 | 7.38 | 6.15 | 3.07 | 1.54 | 0.77 | 0.38 |
| R4_hip | 5.381e-03 | 3.72 | 3.10 | 1.55 | 0.77 | 0.39 | 0.19 |
| R4_knee | 2.711e-03 | 7.38 | 6.15 | 3.07 | 1.54 | 0.77 | 0.38 |

Coarsest substep with every joint below 2: **1/240** (knee ratio 1.54). At the project's 0.02 s the knee ratio is 7.4 -> the explicit torque PD diverges; at 1/240 it is 1.54 (< 2, but the femur recoils so the effective inertia seen by the knee is smaller than the tibia-alone number - the divergence observed at 1/240 is consistent with that); **1/480 (0.77) is the substep the C# torque actuator needs**.
The ArticulationDrive path (the PhysX implicit spring-damper - which is what Isaac's ImplicitActuator is) has no such bound and is stable at the project's 0.02 s.

## 4. Reference-file conventions

root_quat_w_wxyz matches obs[41:43] as **xyzw** (max err xyzw 1.53e-06, wxyz 8.75e+00): the field name says wxyz but the data is xyzw (w last). The copy in this folder renames it root_quat_w_xyzw.
Body height in the reference: min 0.134 m, mean 0.167 m, max 0.203 m (Isaac spawns at 0.18 m; reported standing height 0.141 m).
Mean planar speed over the 200-step recording: 2.76 m/s.

## 5. Colliders

URDF collision shapes are primitives only: body sphere r=0.1; femur cylinder r=0.02 L=0.16; tibia cylinder r=0.015 L=0.26. The URDF Importer turns each cylinder into a scaled convex mesh under a non-uniformly scaled `unnamed` parent; the setup script replaces those with unscaled CapsuleColliders created directly on the link (height = L + r, so the tip contact point matches the flat cylinder rim at the rest angle within ~1 mm).
