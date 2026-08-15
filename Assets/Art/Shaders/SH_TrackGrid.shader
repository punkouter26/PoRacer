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
    }
    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" "Queue" = "Geometry" }

        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode" = "UniversalForward" }

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN
            #pragma multi_compile_fragment _ _SHADOWS_SOFT
            #pragma multi_compile_fog

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            CBUFFER_START(UnityPerMaterial)
                half4 _BaseColor;
                half4 _LineColor;
                float _GridSize;
                float _LineWidth;
                float _TexTileSize;
            CBUFFER_END

            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 positionWS : TEXCOORD0;
                float3 normalWS : TEXCOORD1;
                float fogFactor : TEXCOORD2;
            };

            Varyings vert(Attributes input)
            {
                Varyings output;
                VertexPositionInputs positionInputs = GetVertexPositionInputs(input.positionOS.xyz);
                output.positionCS = positionInputs.positionCS;
                output.positionWS = positionInputs.positionWS;
                output.normalWS = TransformObjectToWorldNormal(input.normalOS);
                output.fogFactor = ComputeFogFactor(positionInputs.positionCS.z);
                return output;
            }

            half4 frag(Varyings input) : SV_Target
            {
                // World-space grid lines, antialiased with screen-space derivatives.
                float2 grid = input.positionWS.xz / _GridSize;
                float2 distToLine = abs(frac(grid + 0.5) - 0.5) * _GridSize;
                float2 fw = fwidth(input.positionWS.xz) + 1e-5;
                float2 lineMask2 = 1.0 - smoothstep(_LineWidth - fw, _LineWidth + fw, distToLine);
                float lineMask = saturate(max(lineMask2.x, lineMask2.y));
                float camDist = distance(_WorldSpaceCameraPos, input.positionWS);
                lineMask *= saturate(1.0 - camDist / 90.0);

                half3 texColor = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap,
                    input.positionWS.xz / _TexTileSize).rgb;
                half3 albedo = lerp(_BaseColor.rgb * texColor, _LineColor.rgb, lineMask);

                float3 normalWS = normalize(input.normalWS);
                float4 shadowCoord = TransformWorldToShadowCoord(input.positionWS);
                Light mainLight = GetMainLight(shadowCoord);
                half3 direct = mainLight.color * mainLight.shadowAttenuation *
                    saturate(dot(normalWS, mainLight.direction));
                half3 ambient = SampleSH(normalWS);
                half3 color = albedo * (direct + ambient);
                color = MixFog(color, input.fogFactor);
                return half4(color, 1.0);
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

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct DepthAttributes
            {
                float4 positionOS : POSITION;
            };

            struct DepthVaryings
            {
                float4 positionCS : SV_POSITION;
            };

            DepthVaryings DepthVert(DepthAttributes input)
            {
                DepthVaryings output;
                output.positionCS = TransformObjectToHClip(input.positionOS.xyz);
                return output;
            }

            half DepthFrag(DepthVaryings input) : SV_Target
            {
                return 0;
            }
            ENDHLSL
        }
    }
}
