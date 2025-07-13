#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

constant static const float kBoundsRadius = 2;
constant static const float kBoundsRadiusSquared = kBoundsRadius*kBoundsRadius;

enum BufferIndex: int32_t
{
    BufferIndexUniforms = 0,
    BufferIndexSplat    = 1,
    BufferIndexOrder    = 2,
};

typedef struct
{
    matrix_float4x4 projectionMatrix;
    matrix_float4x4 viewMatrix;
    uint2 screenSize;
	uint2 Padding;
} Uniforms;

typedef struct
{
    packed_float3 position;
    packed_half4 color;
    packed_half3 covA;
    packed_half3 covB;
} Splat;

typedef struct
{
    float4 position [[position]];
    float2 textureCoordinates;
    float4 color;
} ColorInOut;

float3 calcCovariance2D(float3 viewPos,
                        packed_half3 cov3Da,
                        packed_half3 cov3Db,
                        float4x4 viewMatrix,
                        float4x4 projectionMatrix,
                        uint2 screenSize) {
    float invViewPosZ = 1 / viewPos.z;
    float invViewPosZSquared = invViewPosZ * invViewPosZ;

    float tanHalfFovX = 1 / projectionMatrix[0][0];
    float tanHalfFovY = 1 / projectionMatrix[1][1];
    float limX = 1.3 * tanHalfFovX;
    float limY = 1.3 * tanHalfFovY;
    viewPos.x = clamp(viewPos.x * invViewPosZ, -limX, limX) * viewPos.z;
    viewPos.y = clamp(viewPos.y * invViewPosZ, -limY, limY) * viewPos.z;

    float focalX = screenSize.x * projectionMatrix[0][0] / 2;
    float focalY = screenSize.y * projectionMatrix[1][1] / 2;

    float3x3 J = float3x3(
        focalX * invViewPosZ, 0, 0,
        0, focalY * invViewPosZ, 0,
        -(focalX * viewPos.x) * invViewPosZSquared, -(focalY * viewPos.y) * invViewPosZSquared, 0
    );
    float3x3 W = float3x3(viewMatrix[0].xyz, viewMatrix[1].xyz, viewMatrix[2].xyz);
    float3x3 T = J * W;
    float3x3 Vrk = float3x3(
        cov3Da.x, cov3Da.y, cov3Da.z,
        cov3Da.y, cov3Db.x, cov3Db.y,
        cov3Da.z, cov3Db.y, cov3Db.z
    );
    float3x3 cov = T * Vrk * transpose(T);

    // Apply low-pass filter: every Gaussian should be at least
    // one pixel wide/high. Discard 3rd row and column.
    cov[0][0] += 0.3;
    cov[1][1] += 0.3;
    return float3(cov[0][0], cov[0][1], cov[1][1]);
}

// cov2D is a flattened 2d covariance matrix. Given
// covariance = | a b |
//              | c d |
// (where b == c because the Gaussian covariance matrix is symmetric),
// cov2D = ( a, b, d )
void decomposeCovariance(float3 cov2D, thread float2 &v1, thread float2 &v2) {
    float a = cov2D.x;
    float b = cov2D.y;
    float d = cov2D.z;
    float det = a * d - b * b; // matrix is symmetric, so "c" is same as "b"
    float trace = a + d;

    float mean = 0.5 * trace;
    float dist = max(0.1, sqrt(mean * mean - det)); // based on https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu

    // Eigenvalues
    float lambda1 = mean + dist;
    float lambda2 = mean - dist;

    float2 eigenvector1;
    if (b == 0) {
        eigenvector1 = (a > d) ? float2(1, 0) : float2(0, 1);
    } else {
        eigenvector1 = normalize(float2(b, d - lambda2));
    }

    // Gaussian axes are orthogonal
    float2 eigenvector2 = float2(eigenvector1.y, -eigenvector1.x);

    lambda1 *= 2;
    lambda2 *= 2;

    v1 = eigenvector1 * sqrt(lambda1);
    v2 = eigenvector2 * sqrt(lambda2);
}

vertex ColorInOut splatVertexShader(uint vertexID [[vertex_id]],
                                    uint instanceID [[instance_id]],
                                    //ushort amp_id [[amplification_id]],
                                    constant Splat* splatArray [[ buffer(BufferIndexSplat) ]],
                                    constant metal::uint32_t* orderArray [[ buffer(BufferIndexOrder) ]],
                                    constant Uniforms& uniforms[[ buffer(BufferIndexUniforms) ]]) {
    ColorInOut out;

    Splat splat = splatArray[orderArray[instanceID]];
    float4 viewPosition4 = uniforms.viewMatrix * float4(splat.position, 1);
    float3 viewPosition3 = viewPosition4.xyz;

    float3 cov2D = calcCovariance2D(viewPosition3, splat.covA, splat.covB,
                                    uniforms.viewMatrix, uniforms.projectionMatrix, uniforms.screenSize);

    float2 axis1;
    float2 axis2;
    decomposeCovariance(cov2D, axis1, axis2);

    float4 projectedCenter = uniforms.projectionMatrix * viewPosition4;

    float bounds = 1.2 * projectedCenter.w;
    if (projectedCenter.z < -projectedCenter.w ||
        projectedCenter.x < -bounds ||
        projectedCenter.x > bounds ||
        projectedCenter.y < -bounds ||
        projectedCenter.y > bounds) {
        out.position = float4(0, 0, 2, 1);
        return out;
    }

    float2 textureCoordinates;
    switch (vertexID % 4) {
        case 0: textureCoordinates = float2(0, 0); break;
        case 1: textureCoordinates = float2(0, 1); break;
        case 2: textureCoordinates = float2(1, 0); break;
        case 3: textureCoordinates = float2(1, 1); break;
    }
    float2 vertexRelativePosition = float2((textureCoordinates.x - 0.5) * 2, (textureCoordinates.y - 0.5) * 2);
    float2 screenCenter = projectedCenter.xy / projectedCenter.w;
    float2 screenSizeFloat = float2(uniforms.screenSize.x, uniforms.screenSize.y);
    float2 screenDelta =
        (vertexRelativePosition.x * axis1 +
         vertexRelativePosition.y * axis2)
        * 2
        * kBoundsRadius
        / screenSizeFloat;
    float2 screenVertex = screenCenter + screenDelta;

    out.position = float4(screenVertex.x, screenVertex.y, 0, 1);
    out.textureCoordinates = textureCoordinates;
    out.color = float4(splat.color);
    return out;
}

fragment float4 splatFragmentShader(ColorInOut in [[stage_in]]) {
    float2 v = kBoundsRadius * float2(in.textureCoordinates.x * 2 - 1, in.textureCoordinates.y * 2 - 1);
    float negativeVSquared = -dot(v, v);
    if (negativeVSquared < -kBoundsRadiusSquared) {
        discard_fragment();
    }

    float alpha = saturate(exp(negativeVSquared)) * in.color.a;
    return float4(alpha * in.color.rgb, alpha);
}

