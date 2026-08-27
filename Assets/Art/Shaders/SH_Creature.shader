// Creature surface.
//
// Lighting: half-Lambert base so bodies never go pitch black, a Blinn-Phong
// specular lobe driven by the per-instance smoothness/metallic the spawner sets,
// and a view-angle rim glow that lifts each racer off the ground and keeps its
// tint readable at a distance. Receives and casts main-light shadows.
//
// Surfacing: the project ships no texture assets, so detail is generated in the
// fragment shader. World-space triplanar value noise gives every body a grain
// that does not stretch across the primitive's untextured UVs, and the height
// field it produces is differenced into a perturbed normal, so the grain catches
// the sun instead of only tinting the albedo. _SurfaceId picks the pattern per
// racer (speckle / scales / plates / weave) so a snake does not read as a spider.
//
// Batching: every racer renderer carries a MaterialPropertyBlock for its tint,
// which switches the draw off the SRP Batcher path. The per-instance properties
// below put it on the GPU instancing path instead, so a 100-racer field of shared
// primitive meshes collapses into a handful of instanced draws rather than one
// draw per body part.
Shader "PoRacer/Creature"
{
    Properties
    {
        _BaseMap("Base Map", 2D) = "white" {}
        _BaseColor("Base Color", Color) = (1, 1, 1, 1)
        _SpecColor("Specular Color", Color) = (1, 1, 1, 1)
        _Smoothness("Smoothness", Range(0, 1)) = 0.5
        _Metallic("Metallic", Range(0, 1)) = 0.0
        _RimColor("Rim Color", Color) = (1, 1, 1, 1)
        _RimPower("Rim Power", Float) = 3.0
        _RimStrength("Rim Strength", Float) = 0.55
        // Surface detail. Scale is in metres per tile: creature parts are a few
        // tens of centimetres, so the default gives a few tiles per limb.
        _SurfaceId("Surface Pattern Id", Float) = 0
        _DetailScale("Detail Scale (m)", Float) = 0.22
        _DetailStrength("Detail Albedo Strength", Range(0, 1)) = 0.28
        _DetailNormalStrength("Detail Normal Strength", Range(0, 3)) = 1.1
    }

    SubShader
    {
        Tags
        {
            "RenderType" = "Opaque"
            "RenderPipeline" = "UniversalPipeline"
            "Queue" = "Geometry"
        }

        HLSLINCLUDE
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

        CBUFFER_START(UnityPerMaterial)
            float4 _BaseMap_ST;
            half4 _RimColor;
            half _RimPower;
            half _RimStrength;
            float _DetailScale;
            half _DetailStrength;
            half _DetailNormalStrength;
        CBUFFER_END

        // Per-instance: the spawner writes these through a MaterialPropertyBlock,
        // one value per racer body part. Declared here so they travel in the
        // instancing buffer instead of forcing a separate draw each.
        // Declared as full float: MaterialPropertyBlock uploads 32-bit values, and
        // a half-typed instanced property is a layout mismatch waiting to happen on
        // a backend that packs the buffer differently.
        UNITY_INSTANCING_BUFFER_START(PoRacerPerInstance)
            UNITY_DEFINE_INSTANCED_PROP(float4, _BaseColor)
            UNITY_DEFINE_INSTANCED_PROP(float4, _SpecColor)
            UNITY_DEFINE_INSTANCED_PROP(float, _Smoothness)
            UNITY_DEFINE_INSTANCED_PROP(float, _Metallic)
            UNITY_DEFINE_INSTANCED_PROP(float, _SurfaceId)
        UNITY_INSTANCING_BUFFER_END(PoRacerPerInstance)

        TEXTURE2D(_BaseMap);
        SAMPLER(sampler_BaseMap);

        // --- Procedural detail -------------------------------------------------

        // Cheap 3D value-noise hash. Deterministic, no texture fetch, and stable
        // across platforms because it never relies on float precision past ~1e4.
        float PoRacerHash(float3 cell)
        {
            return frac(sin(dot(cell, float3(127.1, 311.7, 74.7))) * 43758.5453);
        }

        float PoRacerValueNoise(float3 position)
        {
            float3 cell = floor(position);
            float3 f = frac(position);
            // Smoothstep the interpolant so cell edges do not show as creases.
            f = f * f * (3.0 - 2.0 * f);

            float n000 = PoRacerHash(cell + float3(0, 0, 0));
            float n100 = PoRacerHash(cell + float3(1, 0, 0));
            float n010 = PoRacerHash(cell + float3(0, 1, 0));
            float n110 = PoRacerHash(cell + float3(1, 1, 0));
            float n001 = PoRacerHash(cell + float3(0, 0, 1));
            float n101 = PoRacerHash(cell + float3(1, 0, 1));
            float n011 = PoRacerHash(cell + float3(0, 1, 1));
            float n111 = PoRacerHash(cell + float3(1, 1, 1));

            float x00 = lerp(n000, n100, f.x);
            float x10 = lerp(n010, n110, f.x);
            float x01 = lerp(n001, n101, f.x);
            float x11 = lerp(n011, n111, f.x);
            return lerp(lerp(x00, x10, f.y), lerp(x01, x11, f.y), f.z);
        }

        // Height field for the surface grain, in [0,1]. Two octaves is enough at
        // the distances a racer is ever seen from, and keeps the normal-difference
        // below to three evaluations per fragment.
        //
        // surfaceId selects the pattern: 1 = scales (stacked ripples), 2 = plates
        // (hard-edged cells), 3 = weave (crossed threads), anything else = speckle.
        float PoRacerSurfaceHeight(float3 positionWS, half surfaceId)
        {
            float3 p = positionWS / max(_DetailScale, 0.001);
            float grain = PoRacerValueNoise(p) * 0.65 + PoRacerValueNoise(p * 2.17) * 0.35;

            if (surfaceId > 2.5)
            {
                // Weave: two crossed thread runs, softened by the grain.
                float warp = sin(p.x * 3.14159) * 0.5 + 0.5;
                float weft = sin(p.z * 3.14159) * 0.5 + 0.5;
                return saturate(max(warp, weft) * 0.7 + grain * 0.3);
            }
            if (surfaceId > 1.5)
            {
                // Plates: quantised cells with a hard rim, read as chitin.
                float3 cell = floor(p * 0.75);
                float plate = PoRacerHash(cell);
                float edge = length(frac(p * 0.75) - 0.5);
                return saturate(plate * 0.55 + smoothstep(0.28, 0.5, edge) * 0.45);
            }
            if (surfaceId > 0.5)
            {
                // Scales: overlapping ripples running along the body, with the
                // grain breaking up the regularity so it does not look printed.
                float ripple = frac(p.y * 0.8 + grain * 0.35);
                return saturate(smoothstep(0.0, 0.65, ripple) * 0.8 + grain * 0.2);
            }
            return grain;
        }

        // Triplanar blend weights from the world normal, so a primitive with no
        // meaningful UVs still gets a coherent, non-stretched pattern.
        float3 PoRacerTriplanarWeights(float3 normalWS)
        {
            float3 weights = pow(abs(normalWS), 4.0);
            return weights / max(dot(weights, 1.0.xxx), 1e-4);
        }
        ENDHLSL

        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode" = "UniversalForward" }

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE
            #pragma multi_compile _ _SHADOWS_SOFT
            #pragma multi_compile_fog

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 normalWS : TEXCOORD1;
                float3 positionWS : TEXCOORD2;
                float fogFactor : TEXCOORD3;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            Varyings vert(Attributes input)
            {
                Varyings output;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);

                VertexPositionInputs positions = GetVertexPositionInputs(input.positionOS.xyz);
                output.positionCS = positions.positionCS;
                output.positionWS = positions.positionWS;
                output.normalWS = TransformObjectToWorldNormal(input.normalOS);
                output.uv = TRANSFORM_TEX(input.uv, _BaseMap);
                output.fogFactor = ComputeFogFactor(positions.positionCS.z);
                return output;
            }

            half4 frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(input);

                half4 baseColor = (half4)UNITY_ACCESS_INSTANCED_PROP(PoRacerPerInstance, _BaseColor);
                half4 specColor = (half4)UNITY_ACCESS_INSTANCED_PROP(PoRacerPerInstance, _SpecColor);
                half smoothness = (half)UNITY_ACCESS_INSTANCED_PROP(PoRacerPerInstance, _Smoothness);
                half metallic = (half)UNITY_ACCESS_INSTANCED_PROP(PoRacerPerInstance, _Metallic);
                half surfaceId = (half)UNITY_ACCESS_INSTANCED_PROP(PoRacerPerInstance, _SurfaceId);

                half3 albedo = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, input.uv).rgb * baseColor.rgb;
                float3 normalWS = normalize(input.normalWS);
                float3 positionWS = input.positionWS;

                // Height at this point plus two neighbours, differenced into a
                // tangent-free normal perturbation. The offsets are scaled with the
                // detail size so the bumps keep their shape at any _DetailScale.
                float offset = max(_DetailScale, 0.001) * 0.35;
                float height = PoRacerSurfaceHeight(positionWS, surfaceId);
                float heightX = PoRacerSurfaceHeight(positionWS + float3(offset, 0, 0), surfaceId);
                float heightZ = PoRacerSurfaceHeight(positionWS + float3(0, 0, offset), surfaceId);

                // Build the perturbation in world space and reject the component
                // along the geometric normal, so the surface is nudged rather than
                // flipped even where the gradient is steep.
                float3 gradient = float3(height - heightX, 0.0, height - heightZ) * _DetailNormalStrength;
                gradient -= normalWS * dot(gradient, normalWS);
                normalWS = normalize(normalWS + gradient);

                // Triplanar tint of the albedo: the same height field, weighted by
                // the face direction so the grain never smears on a capsule cap.
                float3 weights = PoRacerTriplanarWeights(normalWS);
                float detail = height * weights.y
                    + PoRacerSurfaceHeight(positionWS.yzx, surfaceId) * weights.x
                    + PoRacerSurfaceHeight(positionWS.zxy, surfaceId) * weights.z;
                albedo *= lerp(1.0h, (half)(0.6 + detail * 0.8), _DetailStrength);

                float4 shadowCoord = TransformWorldToShadowCoord(positionWS);
                Light mainLight = GetMainLight(shadowCoord);
                half shadow = mainLight.shadowAttenuation;

                half halfLambert = dot(normalWS, mainLight.direction) * 0.5h + 0.5h;
                half3 direct = mainLight.color * (halfLambert * halfLambert) * shadow;
                half3 ambient = SampleSH(normalWS);

                float3 viewDir = normalize(GetWorldSpaceViewDir(positionWS));

                // Blinn-Phong lobe. Metallic tints the highlight toward the body
                // colour the way a metal does; a dielectric keeps it white.
                half3 halfVector = normalize(mainLight.direction + viewDir);
                half specularPower = exp2(smoothness * 10.0h + 1.0h);
                half specularTerm = pow(saturate(dot(normalWS, halfVector)), specularPower);
                half3 specularTint = lerp(specColor.rgb, specColor.rgb * albedo, metallic);
                half3 specular = specularTint * specularTerm * smoothness * shadow;

                half3 color = albedo * (direct + ambient) + specular * mainLight.color;

                half rim = pow(saturate(1.0h - dot(normalWS, viewDir)), _RimPower) * _RimStrength;
                color += rim * _RimColor.rgb * (albedo + 0.2h);

                color = MixFog(color, input.fogFactor);
                return half4(color, 1.0h);
            }
            ENDHLSL
        }

        // Explicit shadow and depth passes rather than UsePass on the URP Lit
        // shader: they need the instancing pragma too, or every shadow-casting
        // body part falls back to its own draw and undoes the batching above.
        Pass
        {
            Name "ShadowCaster"
            Tags { "LightMode" = "ShadowCaster" }

            ZWrite On
            ZTest LEqual
            ColorMask 0
            Cull Back

            HLSLPROGRAM
            #pragma vertex ShadowVert
            #pragma fragment ShadowFrag
            #pragma multi_compile_instancing
            #pragma multi_compile_vertex _ _CASTING_PUNCTUAL_LIGHT_SHADOW

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Shadows.hlsl"

            float3 _LightDirection;
            float3 _LightPosition;

            struct ShadowAttributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct ShadowVaryings
            {
                float4 positionCS : SV_POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            ShadowVaryings ShadowVert(ShadowAttributes input)
            {
                ShadowVaryings output;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);

                float3 positionWS = TransformObjectToWorld(input.positionOS.xyz);
                float3 normalWS = TransformObjectToWorldNormal(input.normalOS);

                #if defined(_CASTING_PUNCTUAL_LIGHT_SHADOW)
                    float3 lightDirectionWS = normalize(_LightPosition - positionWS);
                #else
                    float3 lightDirectionWS = _LightDirection;
                #endif

                float4 positionCS = TransformWorldToHClip(
                    ApplyShadowBias(positionWS, normalWS, lightDirectionWS));

                #if UNITY_REVERSED_Z
                    positionCS.z = min(positionCS.z, UNITY_NEAR_CLIP_VALUE);
                #else
                    positionCS.z = max(positionCS.z, UNITY_NEAR_CLIP_VALUE);
                #endif

                output.positionCS = positionCS;
                return output;
            }

            half4 ShadowFrag(ShadowVaryings input) : SV_Target
            {
                return 0;
            }
            ENDHLSL
        }

        Pass
        {
            Name "DepthOnly"
            Tags { "LightMode" = "DepthOnly" }

            ZWrite On
            ColorMask R
            Cull Back

            HLSLPROGRAM
            #pragma vertex DepthVert
            #pragma fragment DepthFrag
            #pragma multi_compile_instancing

            struct DepthAttributes
            {
                float4 positionOS : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct DepthVaryings
            {
                float4 positionCS : SV_POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            DepthVaryings DepthVert(DepthAttributes input)
            {
                DepthVaryings output;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);
                output.positionCS = TransformObjectToHClip(input.positionOS.xyz);
                return output;
            }

            half DepthFrag(DepthVaryings input) : SV_Target
            {
                return 0;
            }
            ENDHLSL
        }

        // DepthNormals feeds screen-space ambient occlusion; without it the SSAO
        // renderer feature has no normal buffer for these bodies and they are
        // excluded from the effect.
        Pass
        {
            Name "DepthNormals"
            Tags { "LightMode" = "DepthNormals" }

            ZWrite On
            Cull Back

            HLSLPROGRAM
            #pragma vertex DepthNormalsVert
            #pragma fragment DepthNormalsFrag
            #pragma multi_compile_instancing

            struct DepthNormalsAttributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct DepthNormalsVaryings
            {
                float4 positionCS : SV_POSITION;
                float3 normalWS : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            DepthNormalsVaryings DepthNormalsVert(DepthNormalsAttributes input)
            {
                DepthNormalsVaryings output;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);
                output.positionCS = TransformObjectToHClip(input.positionOS.xyz);
                output.normalWS = TransformObjectToWorldNormal(input.normalOS);
                return output;
            }

            half4 DepthNormalsFrag(DepthNormalsVaryings input) : SV_Target
            {
                return half4(normalize(input.normalWS) * 0.5 + 0.5, 0.0);
            }
            ENDHLSL
        }
    }

    FallBack "Universal Render Pipeline/Lit"
}
