// Track ground.
//
// The lane grid stays as it was: world-space lines, derivative-antialiased, fading
// out with distance so the horizon does not turn into moire.
//
// On top of it the surface now generates its own detail. The project ships no
// ground textures, so a two-octave value-noise height field supplies both a
// coarse tint variation and a differenced normal, which is what gives the sun
// something to rake across on the Lumpy and Swamp maps. Slope is read off the
// geometric normal: flanks of hills pick up an exposed-earth tint that flat
// ground does not, so procedurally built terrain reads as terrain.
//
// _PoRacerWetness is a global, written per map by PostFxView. It darkens the
// albedo and opens up a specular lobe, turning the same mesh from dry dirt into
// wet swamp without a second material.
Shader "PoRacer/TrackGrid"
{
    Properties
    {
        _BaseColor ("Base Color", Color) = (0.16, 0.17, 0.2, 1)
        _LineColor ("Line Color", Color) = (0.95, 0.55, 0.2, 1)
        _GridSize ("Grid Size (m)", Float) = 2.0
        _LineWidth ("Line Width (m)", Float) = 0.04
        // Sampled in world space (1 tile per _TexTileSize m), so meshes need no UVs.
        _BaseMap ("Base Map", 2D) = "white" {}
        _TexTileSize ("Texture Tile Size (m)", Float) = 4.0
        _DetailScale ("Detail Scale (m)", Float) = 1.6
        _DetailStrength ("Detail Albedo Strength", Range(0, 1)) = 0.35
        _DetailNormalStrength ("Detail Normal Strength", Range(0, 4)) = 1.6
        _SlopeColor ("Exposed Slope Color", Color) = (0.32, 0.26, 0.19, 1)
    }
    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" "Queue" = "Geometry" }

        HLSLINCLUDE
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

        CBUFFER_START(UnityPerMaterial)
            half4 _BaseColor;
            half4 _LineColor;
            half4 _SlopeColor;
            float _GridSize;
            float _LineWidth;
            float _TexTileSize;
            float _DetailScale;
            half _DetailStrength;
            half _DetailNormalStrength;
        CBUFFER_END

        // Global, not a material property: one write per map change covers every
        // ground chunk the track builder spawned. Declared outside UnityPerMaterial
        // so the shader stays SRP Batcher compatible.
        half _PoRacerWetness;

        float TrackHash(float2 cell)
        {
            return frac(sin(dot(cell, float2(127.1, 311.7))) * 43758.5453);
        }

        float TrackValueNoise(float2 position)
        {
            float2 cell = floor(position);
            float2 f = frac(position);
            f = f * f * (3.0 - 2.0 * f);
            float n00 = TrackHash(cell + float2(0, 0));
            float n10 = TrackHash(cell + float2(1, 0));
            float n01 = TrackHash(cell + float2(0, 1));
            float n11 = TrackHash(cell + float2(1, 1));
            return lerp(lerp(n00, n10, f.x), lerp(n01, n11, f.x), f.y);
        }

        // Ground grain in [0,1]. Two octaves: a metre-scale undulation for the
        // tint, and a finer one that the normal difference below turns into tooth.
        float TrackSurfaceHeight(float2 positionXZ)
        {
            float2 p = positionXZ / max(_DetailScale, 0.01);
            return TrackValueNoise(p) * 0.62 + TrackValueNoise(p * 3.3) * 0.38;
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
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN
            #pragma multi_compile_fragment _ _SHADOWS_SOFT
            #pragma multi_compile_fog

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 positionWS : TEXCOORD0;
                float3 normalWS : TEXCOORD1;
                float fogFactor : TEXCOORD2;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            Varyings vert(Attributes input)
            {
                Varyings output;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);
                VertexPositionInputs positionInputs = GetVertexPositionInputs(input.positionOS.xyz);
                output.positionCS = positionInputs.positionCS;
                output.positionWS = positionInputs.positionWS;
                output.normalWS = TransformObjectToWorldNormal(input.normalOS);
                output.fogFactor = ComputeFogFactor(positionInputs.positionCS.z);
                return output;
            }

            half4 frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(input);

                float3 positionWS = input.positionWS;

                // World-space grid lines, antialiased with screen-space derivatives.
                float2 grid = positionWS.xz / _GridSize;
                float2 distToLine = abs(frac(grid + 0.5) - 0.5) * _GridSize;
                float2 fw = fwidth(positionWS.xz) + 1e-5;
                float2 lineMask2 = 1.0 - smoothstep(_LineWidth - fw, _LineWidth + fw, distToLine);
                float lineMask = saturate(max(lineMask2.x, lineMask2.y));
                float camDist = distance(_WorldSpaceCameraPos, positionWS);
                lineMask *= saturate(1.0 - camDist / 90.0);

                half3 texColor = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap,
                    positionWS.xz / _TexTileSize).rgb;
                half3 albedo = _BaseColor.rgb * texColor;

                // Procedural grain, and the same field differenced into a normal so
                // the ground has relief the sun can pick out.
                float offset = max(_DetailScale, 0.01) * 0.25;
                float height = TrackSurfaceHeight(positionWS.xz);
                float heightX = TrackSurfaceHeight(positionWS.xz + float2(offset, 0));
                float heightZ = TrackSurfaceHeight(positionWS.xz + float2(0, offset));
                albedo *= lerp(1.0h, (half)(0.62 + height * 0.76), _DetailStrength);

                float3 normalWS = normalize(input.normalWS);
                float3 gradient = float3(height - heightX, 0.0, height - heightZ) * _DetailNormalStrength;
                gradient -= normalWS * dot(gradient, normalWS);
                normalWS = normalize(normalWS + gradient);

                // Steep faces of procedurally raised terrain show bare earth; flat
                // ground keeps the track colour.
                half slope = saturate((1.0h - (half)normalWS.y) * 2.2h);
                albedo = lerp(albedo, _SlopeColor.rgb, slope * 0.65h);

                // Wet ground drinks light and returns a hard highlight instead.
                half wetness = saturate(_PoRacerWetness);
                albedo *= lerp(1.0h, 0.55h, wetness);
                albedo = lerp(albedo, _LineColor.rgb, lineMask);

                float4 shadowCoord = TransformWorldToShadowCoord(positionWS);
                Light mainLight = GetMainLight(shadowCoord);
                half3 direct = mainLight.color * mainLight.shadowAttenuation *
                    saturate(dot(normalWS, mainLight.direction));
                half3 ambient = SampleSH(normalWS);
                half3 color = albedo * (direct + ambient);

                if (wetness > 0.001h)
                {
                    float3 viewDir = normalize(GetWorldSpaceViewDir(positionWS));
                    half3 halfVector = normalize(mainLight.direction + viewDir);
                    half sheen = pow(saturate(dot(normalWS, halfVector)), 90.0h);
                    color += mainLight.color * sheen * wetness * mainLight.shadowAttenuation * 0.9h;
                }

                color = MixFog(color, input.fogFactor);
                return half4(color, 1.0);
            }
            ENDHLSL
        }

        Pass
        {
            Name "ShadowCaster"
            Tags { "LightMode" = "ShadowCaster" }

            ZWrite On
            ZTest LEqual
            ColorMask 0

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

        // Required by the SSAO renderer feature: without a normal buffer entry the
        // ground is excluded from ambient occlusion, and creatures end up with
        // contact darkening that has nothing to sit against.
        Pass
        {
            Name "DepthNormals"
            Tags { "LightMode" = "DepthNormals" }
            ZWrite On

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
                float3 positionWS : TEXCOORD1;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            DepthNormalsVaryings DepthNormalsVert(DepthNormalsAttributes input)
            {
                DepthNormalsVaryings output;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);
                VertexPositionInputs positionInputs = GetVertexPositionInputs(input.positionOS.xyz);
                output.positionCS = positionInputs.positionCS;
                output.positionWS = positionInputs.positionWS;
                output.normalWS = TransformObjectToWorldNormal(input.normalOS);
                return output;
            }

            half4 DepthNormalsFrag(DepthNormalsVaryings input) : SV_Target
            {
                // The same perturbation the lit pass applies, so occlusion follows
                // the visible relief rather than the flat collision mesh.
                float3 normalWS = normalize(input.normalWS);
                float offset = max(_DetailScale, 0.01) * 0.25;
                float height = TrackSurfaceHeight(input.positionWS.xz);
                float heightX = TrackSurfaceHeight(input.positionWS.xz + float2(offset, 0));
                float heightZ = TrackSurfaceHeight(input.positionWS.xz + float2(0, offset));
                float3 gradient = float3(height - heightX, 0.0, height - heightZ) * _DetailNormalStrength;
                gradient -= normalWS * dot(gradient, normalWS);
                normalWS = normalize(normalWS + gradient);
                return half4(normalWS * 0.5 + 0.5, 0.0);
            }
            ENDHLSL
        }
    }
}
