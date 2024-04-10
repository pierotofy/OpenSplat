#include <metal_stdlib>

using namespace metal;

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define CHANNELS 3

constant float SH_C0 = 0.28209479177387814f;
constant float SH_C1 = 0.4886025119029199f;
constant float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f};
constant float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f};
constant float SH_C4[] = {
    2.5033429417967046f,
    -1.7701307697799304,
    0.9461746957575601f,
    -0.6690465435572892f,
    0.10578554691520431f,
    -0.6690465435572892f,
    0.47308734787878004f,
    -1.7701307697799304f,
    0.6258357354491761f};

inline uint num_sh_bases(const uint degree) {
    if (degree == 0)
        return 1;
    if (degree == 1)
        return 4;
    if (degree == 2)
        return 9;
    if (degree == 3)
        return 16;
    return 25;
}

inline float ndc2pix(const float x, const float W, const float cx) {
    return 0.5f * W * x + cx - 0.5;
}

inline void get_bbox(
    const float2 center,
    const float2 dims,
    const int3 img_size,
    thread uint2 &bb_min,
    thread uint2 &bb_max
) {
    // get bounding box with center and dims, within bounds
    // bounding box coords returned in tile coords, inclusive min, exclusive max
    // clamp between 0 and tile bounds
    bb_min.x = min(max(0, (int)(center.x - dims.x)), img_size.x);
    bb_max.x = min(max(0, (int)(center.x + dims.x + 1)), img_size.x);
    bb_min.y = min(max(0, (int)(center.y - dims.y)), img_size.y);
    bb_max.y = min(max(0, (int)(center.y + dims.y + 1)), img_size.y);
}

inline void get_tile_bbox(
    const float2 pix_center,
    const float pix_radius,
    const int3 tile_bounds,
    thread uint2 &tile_min,
    thread uint2 &tile_max
) {
    // gets gaussian dimensions in tile space, i.e. the span of a gaussian in
    // tile_grid (image divided into tiles)
    float2 tile_center = {
        pix_center.x / (float)BLOCK_X, pix_center.y / (float)BLOCK_Y
    };
    float2 tile_radius = {
        pix_radius / (float)BLOCK_X, pix_radius / (float)BLOCK_Y
    };
    get_bbox(tile_center, tile_radius, tile_bounds, tile_min, tile_max);
}

// helper for applying R * p + T, expect mat to be ROW MAJOR
inline float3 transform_4x3(constant float *mat, const float3 p) {
    float3 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
    };
    return out;
}

// helper to apply 4x4 transform to 3d vector, return homo coords
// expects mat to be ROW MAJOR
inline float4 transform_4x4(constant float *mat, const float3 p) {
    float4 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
        mat[12] * p.x + mat[13] * p.y + mat[14] * p.z + mat[15],
    };
    return out;
}

inline float3x3 quat_to_rotmat(const float4 quat) {
    // quat to rotation matrix
    float s = rsqrt(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
    );
    float w = quat.x * s;
    float x = quat.y * s;
    float y = quat.z * s;
    float z = quat.w * s;

    // metal matrices are column-major
    return float3x3(
        1.f - 2.f * (y * y + z * z),
        2.f * (x * y + w * z),
        2.f * (x * z - w * y),
        2.f * (x * y - w * z),
        1.f - 2.f * (x * x + z * z),
        2.f * (y * z + w * x),
        2.f * (x * z + w * y),
        2.f * (y * z - w * x),
        1.f - 2.f * (x * x + y * y)
    );
}

// device helper for culling near points
inline bool clip_near_plane(
    const float3 p, 
    constant float *viewmat, 
    thread float3 &p_view, 
    float thresh
) {
    p_view = transform_4x3(viewmat, p);
    if (p_view.z <= thresh) {
        return true;
    }
    return false;
}

inline float3x3 scale_to_mat(const float3 scale, const float glob_scale) {
    float3x3 S = float3x3(1.f);
    S[0][0] = glob_scale * scale.x;
    S[1][1] = glob_scale * scale.y;
    S[2][2] = glob_scale * scale.z;
    return S;
}

// device helper to get 3D covariance from scale and quat parameters
inline void scale_rot_to_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, device float *cov3d
) {
    // printf("quat %.2f %.2f %.2f %.2f\n", quat.x, quat.y, quat.z, quat.w);
    float3x3 R = quat_to_rotmat(quat);
    // printf("R %.2f %.2f %.2f\n", R[0][0], R[1][1], R[2][2]);
    float3x3 S = scale_to_mat(scale, glob_scale);
    // printf("S %.2f %.2f %.2f\n", S[0][0], S[1][1], S[2][2]);

    float3x3 M = R * S;
    float3x3 tmp = M * transpose(M);
    // printf("tmp %.2f %.2f %.2f\n", tmp[0][0], tmp[1][1], tmp[2][2]);

    // save upper right because symmetric
    cov3d[0] = tmp[0][0];
    cov3d[1] = tmp[0][1];
    cov3d[2] = tmp[0][2];
    cov3d[3] = tmp[1][1];
    cov3d[4] = tmp[1][2];
    cov3d[5] = tmp[2][2];
}

// device helper to approximate projected 2d cov from 3d mean and cov
float3 project_cov3d_ewa(
    thread float3& mean3d,
    device float* cov3d,
    constant float* viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy
) {
    // clip the
    // we expect row major matrices as input, metal uses column major
    // upper 3x3 submatrix
    float3x3 W = float3x3(
        viewmat[0],
        viewmat[4],
        viewmat[8],
        viewmat[1],
        viewmat[5],
        viewmat[9],
        viewmat[2],
        viewmat[6],
        viewmat[10]
    );
    float3 p = float3(viewmat[3], viewmat[7], viewmat[11]);
    float3 t = W * float3(mean3d.x, mean3d.y, mean3d.z) + p;

    // clip so that the covariance
    float lim_x = 1.3 * tan_fovx;
    float lim_y = 1.3 * tan_fovy;
    t.x = t.z * min(lim_x, max(-lim_x, t.x / t.z));
    t.y = t.z * min(lim_y, max(-lim_y, t.y / t.z));

    float rz = 1.f / t.z;
    float rz2 = rz * rz;

    // column major
    // we only care about the top 2x2 submatrix
    float3x3 J = float3x3(
        fx * rz,
        0.f,
        0.f,
        0.f,
        fy * rz,
        0.f,
        -fx * t.x * rz2,
        -fy * t.y * rz2,
        0.f
    );
    float3x3 T = J * W;

    float3x3 V = float3x3(
        cov3d[0],
        cov3d[1],
        cov3d[2],
        cov3d[1],
        cov3d[3],
        cov3d[4],
        cov3d[2],
        cov3d[4],
        cov3d[5]
    );

    float3x3 cov = T * V * transpose(T);

    // add a little blur along axes and save upper triangular elements
    return float3(float(cov[0][0]) + 0.3f, float(cov[0][1]), float(cov[1][1]) + 0.3f);
}

inline bool compute_cov2d_bounds(
    const float3 cov2d, 
    thread float3 &conic, 
    thread float &radius
) {
    // find eigenvalues of 2d covariance matrix
    // expects upper triangular values of cov matrix as float3
    // then compute the radius and conic dimensions
    // the conic is the inverse cov2d matrix, represented here with upper
    // triangular values.
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.f)
        return false;
    float inv_det = 1.f / det;

    // inverse of 2x2 cov2d matrix
    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;

    float b = 0.5f * (cov2d.x + cov2d.z);
    float v1 = b + sqrt(max(0.1f, b * b - det));
    float v2 = b - sqrt(max(0.1f, b * b - det));
    // take 3 sigma of covariance
    radius = ceil(3.f * sqrt(max(v1, v2)));
    return true;
}

inline float2 project_pix(
    constant float *mat, const float3 p, const int3 img_size, const float2 pp
) {
    // ROW MAJOR mat
    float4 p_hom = transform_4x4(mat, p);
    float rw = 1.f / (p_hom.w + 1e-6f);
    float3 p_proj = {p_hom.x * rw, p_hom.y * rw, p_hom.z * rw};
    return {
        ndc2pix(p_proj.x, img_size.x, pp.x), ndc2pix(p_proj.y, img_size.y, pp.y)
    };
}

/* 
    !!!!IMPORTANT!!!
    Metal does not support packed arrays of vectorized types like int2, float2, float3, etc.
    and instead pads the elements of arrays of these types to fixed alignments. 
    Use the below functions to read and write from packed arrays of these types.
*/

inline int2 read_packed_int2(constant int* arr, int idx) {
    return int2(arr[2*idx], arr[2*idx+1]);
}

inline void write_packed_int2(device int* arr, int idx, int2 val) {
    arr[2*idx] = val.x;
    arr[2*idx+1] = val.y;
}

inline float2 read_packed_float2(constant float* arr, int idx) {
    return float2(arr[2*idx], arr[2*idx+1]);
}

inline void write_packed_float2(device float* arr, int idx, float2 val) {
    arr[2*idx] = val.x;
    arr[2*idx+1] = val.y;
}

inline int3 read_packed_int3(constant int* arr, int idx) {
    return int3(arr[3*idx], arr[3*idx+1], arr[3*idx+2]);
}

inline void write_packed_int3(device int* arr, int idx, int3 val) {
    arr[3*idx] = val.x;
    arr[3*idx+1] = val.y;
    arr[3*idx+2] = val.z;
}

inline float3 read_packed_float3(constant float* arr, int idx) {
    return float3(arr[3*idx], arr[3*idx+1], arr[3*idx+2]);
}

inline void write_packed_float3(device float* arr, int idx, float3 val) {
    arr[3*idx] = val.x;
    arr[3*idx+1] = val.y;
    arr[3*idx+2] = val.z;
}

inline float4 read_packed_float4(constant float* arr, int idx) {
    return float4(arr[4*idx], arr[4*idx+1], arr[4*idx+2], arr[4*idx+3]);
}

inline void write_packed_float4(device float* arr, int idx, float4 val) {
    arr[4*idx] = val.x;
    arr[4*idx+1] = val.y;
    arr[4*idx+2] = val.z;
    arr[4*idx+3] = val.w;
}

// kernel function for projecting each gaussian on device
// each thread processes one gaussian 
kernel void project_gaussians_forward_kernel(
    constant int& num_points,
    constant float* means3d, // float3
    constant float* scales, // float3
    constant float& glob_scale,
    constant float* quats, // float4
    constant float* viewmat,
    constant float* projmat,
    constant float4& intrins,
    constant int3& img_size,
    constant int3& tile_bounds,
    constant float& clip_thresh,
    device float* covs3d,
    device float* xys, // float2
    device float* depths,
    device int* radii,
    device float* conics, // float3
    device int32_t* num_tiles_hit,
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_points) {
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    float3 p_world = means3d[idx*3];
    float3 p_view;
    if (clip_near_plane(p_world, viewmat, p_view, clip_thresh)) {
        return;
    }

    // compute the projected covariance
    float3 scale = read_packed_float3(scales, idx);
    float4 quat = read_packed_float4(quats, idx);
    device float *cur_cov3d = &(covs3d[6 * idx]);
    scale_rot_to_cov3d(scale, glob_scale, quat, cur_cov3d);

    // project to 2d with ewa approximation
    float fx = intrins.x;
    float fy = intrins.y;
    float cx = intrins.z;
    float cy = intrins.w;
    float tan_fovx = 0.5 * img_size.x / fx;
    float tan_fovy = 0.5 * img_size.y / fy;
    float3 cov2d = project_cov3d_ewa(
        p_world, cur_cov3d, viewmat, fx, fy, tan_fovx, tan_fovy
    );

    float3 conic;
    float radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    if (!ok) {
        return; // zero determinant
    }
    write_packed_float3(conics, idx, conic);

    // compute the projected mean
    float2 center = project_pix(projmat, p_world, img_size, {cx, cy});
    uint2 tile_min, tile_max;
    get_tile_bbox(center, radius, tile_bounds, tile_min, tile_max);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) {
        return;
    }

    num_tiles_hit[idx] = tile_area;
    depths[idx] = p_view.z;
    radii[idx] = (int)radius;
    write_packed_float2(xys, idx, center);
}

kernel void rasterize_forward_kernel(
    constant int3& tile_bounds,
    constant int3& img_size,
    constant uint& channels,
    constant int32_t* gaussian_ids_sorted,
    constant int* tile_bins, // int2
    constant float* xys, // float2
    constant float* conics, // float3
    constant float* colors,
    constant float* opacities,
    device float* final_Ts,
    device int* final_index,
    device float* out_img,
    constant float* background,
    constant uint2& blockDim, 
    uint2 blockIdx [[threadgroup_position_in_grid]],
    uint2 threadIdx [[thread_position_in_threadgroup]]
) {
    // current naive implementation where tile data loading is redundant
    // TODO tile data should be shared between tile threads
    int32_t tile_id = blockIdx.y * tile_bounds.x + blockIdx.x;
    int32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * img_size.x + j;

    // return if out of bounds
    if (i >= img_size.y || j >= img_size.x) {
        return;
    }

    // which gaussians to look through in this tile
    int2 range = read_packed_int2(tile_bins, tile_id);
    float T = 1.f;

    // iterate over all gaussians and apply rendering EWA equation (e.q. 2 from
    // paper)
    int idx;
    for (idx = range.x; idx < range.y; ++idx) {
        const int32_t g = gaussian_ids_sorted[idx];
        const float3 conic = read_packed_float3(conics, g);
        const float2 center = read_packed_float2(xys, g);
        const float2 delta = {center.x - px, center.y - py};

        // Mahalanobis distance (here referred to as sigma) measures how many
        // standard deviations away distance delta is. sigma = -0.5(d.T * conic
        // * d)
        const float sigma =
            0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
            conic.y * delta.x * delta.y;
        if (sigma < 0.f) {
            continue;
        }
        const float opac = opacities[g];

        const float alpha = min(0.999f, opac * exp(-sigma));

        // break out conditions
        if (alpha < 1.f / 255.f) {
            continue;
        }
        const float next_T = T * (1.f - alpha);
        if (next_T <= 1e-4f) {
            // we want to render the last gaussian that contributes and note
            // that here idx > range.x so we don't underflow
            idx -= 1;
            break;
        }
        const float vis = alpha * T;
        for (int c = 0; c < channels; ++c) {
            out_img[channels * pix_id + c] += colors[channels * g + c] * vis;
        }
        T = next_T;
    }
    final_Ts[pix_id] = T; // transmittance at last gaussian in this pixel
    final_index[pix_id] =
        (idx == range.y)
            ? idx - 1
            : idx; // index of in bin of last gaussian in this pixel
    for (int c = 0; c < channels; ++c) {
        out_img[channels * pix_id + c] += T * background[c];
    }
}

void sh_coeffs_to_color(
    const uint degree,
    const float3 viewdir,
    constant float *coeffs,
    device float *colors
) {
    // Expects v_colors to be len CHANNELS
    // and v_coeffs to be num_bases * CHANNELS
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] = SH_C0 * coeffs[c];
    }
    if (degree < 1) {
        return;
    }

    float norm = sqrt(
        viewdir.x * viewdir.x + viewdir.y * viewdir.y + viewdir.z * viewdir.z
    );
    float x = viewdir.x / norm;
    float y = viewdir.y / norm;
    float z = viewdir.z / norm;

    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;
    // expects CHANNELS * num_bases coefficients
    // supports up to num_bases = 25
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] += SH_C1 * (-y * coeffs[1 * CHANNELS + c] +
                              z * coeffs[2 * CHANNELS + c] -
                              x * coeffs[3 * CHANNELS + c]);
        if (degree < 2) {
            continue;
        }
        colors[c] +=
            (SH_C2[0] * xy * coeffs[4 * CHANNELS + c] +
             SH_C2[1] * yz * coeffs[5 * CHANNELS + c] +
             SH_C2[2] * (2.f * zz - xx - yy) * coeffs[6 * CHANNELS + c] +
             SH_C2[3] * xz * coeffs[7 * CHANNELS + c] +
             SH_C2[4] * (xx - yy) * coeffs[8 * CHANNELS + c]);
        if (degree < 3) {
            continue;
        }
        colors[c] +=
            (SH_C3[0] * y * (3.f * xx - yy) * coeffs[9 * CHANNELS + c] +
             SH_C3[1] * xy * z * coeffs[10 * CHANNELS + c] +
             SH_C3[2] * y * (4.f * zz - xx - yy) * coeffs[11 * CHANNELS + c] +
             SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy) *
                 coeffs[12 * CHANNELS + c] +
             SH_C3[4] * x * (4.f * zz - xx - yy) * coeffs[13 * CHANNELS + c] +
             SH_C3[5] * z * (xx - yy) * coeffs[14 * CHANNELS + c] +
             SH_C3[6] * x * (xx - 3.f * yy) * coeffs[15 * CHANNELS + c]);
        if (degree < 4) {
            continue;
        }
        colors[c] +=
            (SH_C4[0] * xy * (xx - yy) * coeffs[16 * CHANNELS + c] +
             SH_C4[1] * yz * (3.f * xx - yy) * coeffs[17 * CHANNELS + c] +
             SH_C4[2] * xy * (7.f * zz - 1.f) * coeffs[18 * CHANNELS + c] +
             SH_C4[3] * yz * (7.f * zz - 3.f) * coeffs[19 * CHANNELS + c] +
             SH_C4[4] * (zz * (35.f * zz - 30.f) + 3.f) *
                 coeffs[20 * CHANNELS + c] +
             SH_C4[5] * xz * (7.f * zz - 3.f) * coeffs[21 * CHANNELS + c] +
             SH_C4[6] * (xx - yy) * (7.f * zz - 1.f) *
                 coeffs[22 * CHANNELS + c] +
             SH_C4[7] * xz * (xx - 3.f * yy) * coeffs[23 * CHANNELS + c] +
             SH_C4[8] * (xx * (xx - 3.f * yy) - yy * (3.f * xx - yy)) *
                 coeffs[24 * CHANNELS + c]);
    }
}

kernel void compute_sh_forward_kernel(
    constant uint& num_points,
    constant uint& degree,
    constant uint& degrees_to_use,
    constant float* viewdirs, // float3
    constant float* coeffs,
    device float* colors,
    uint idx [[thread_position_in_threadgroup]]
) {
    if (idx >= num_points) {
        return;
    }
    const uint num_channels = 3;
    uint num_bases = num_sh_bases(degree);
    uint idx_sh = num_bases * num_channels * idx;
    uint idx_col = num_channels * idx;

    sh_coeffs_to_color(
        degrees_to_use, read_packed_float3(viewdirs, idx), &(coeffs[idx_sh]), &(colors[idx_col])
    );
}