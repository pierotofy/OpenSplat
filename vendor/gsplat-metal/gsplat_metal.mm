#import "bindings.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

struct MetalContext {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    dispatch_queue_t d_queue;

    id<MTLComputePipelineState> nd_rasterize_forward_kernel_cpso;
    id<MTLComputePipelineState> nd_rasterize_backward_kernel_cpso;
    id<MTLComputePipelineState> rasterize_forward_kernel_cpso;
    id<MTLComputePipelineState> rasterize_backward_kernel_cpso;
    id<MTLComputePipelineState> project_gaussians_forward_kernel_cpso;
    id<MTLComputePipelineState> project_gaussians_backward_kernel_cpso;
    id<MTLComputePipelineState> compute_sh_forward_kernel_cpso;
    id<MTLComputePipelineState> compute_sh_backward_kernel_cpso;
    id<MTLComputePipelineState> compute_cov2d_bounds_kernel_cpso;
    id<MTLComputePipelineState> map_gaussian_to_intersects_kernel_cpso;
    id<MTLComputePipelineState> get_tile_bin_edges_kernel_cpso;
};

// This function is used in both host and device code
// TODO(achan): Do I need to make this callable from the metal device?
unsigned num_sh_bases(const unsigned degree) {
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

// This empty class lets us query for files relative to this file's bundle path using NSBundle bundleForClass hack
@interface DummyClassForPathHack : NSObject
@end
@implementation DummyClassForPathHack
@end

MetalContext* init_gsplat_metal_context() {
    MetalContext* ctx = (MetalContext*)malloc(sizeof(MetalContext));
    // Retrieve the default Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    // Configure context
    ctx->device = device;
    ctx->queue  = [ctx->device newCommandQueue];
    ctx->d_queue = torch::mps::get_dispatch_queue();

    NSError *error = nil;

    id<MTLLibrary> metal_library = nil;
    NSBundle * bundle = [NSBundle bundleForClass:[DummyClassForPathHack class]];
    NSString * path_lib = [bundle pathForResource:@"default" ofType:@"metallib"];

    if (path_lib != nil) {
        // pre-compiled library found
        NSURL * libURL = [NSURL fileURLWithPath:path_lib];
        printf("%s: loading '%s'\n", __func__, [path_lib UTF8String]);

        metal_library = [ctx->device newLibraryWithURL:libURL error:&error];
        if (error) {
            printf("%s: error: %s\n", __func__, [[error description] UTF8String]);
            return NULL;
        }
        printf("%s: loaded '%s', functions: %s\n", __func__, [path_lib UTF8String], [[[metal_library functionNames] componentsJoinedByString:@", "] UTF8String]);
    } else {
        printf("%s: default.metallib not found, loading from source\n", __func__);

        NSString * source_path = [[@ __FILE__ stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"ggml-metal.metal"];
        printf("%s: loading '%s'\n", __func__, [source_path UTF8String]);

        NSString * src = [NSString stringWithContentsOfFile:source_path encoding:NSUTF8StringEncoding error:&error];
        if (error) {
            printf("%s: error: %s\n", __func__, [[error description] UTF8String]);
            return NULL;
        }

        @autoreleasepool {
            // dictionary of preprocessor macros
            NSMutableDictionary * prep = [NSMutableDictionary dictionary];

            MTLCompileOptions* options = [MTLCompileOptions new];
            options.preprocessorMacros = prep;

            metal_library = [ctx->device newLibraryWithSource:src options:options error:&error];
            if (error) {
                printf("%s: error: %s\n", __func__, [[error description] UTF8String]);
                return NULL;
            }
        }
    }

#define GSPLAT_METAL_ADD_KERNEL(NAME) \
    { \
        id<MTLFunction> metal_function = [metal_library newFunctionWithName:@#NAME]; \
        printf("%s: load function %s with label: %s\n", __func__, #NAME, [[metal_function label] UTF8String]); \
        ctx->NAME ## _cpso = [ctx->device newComputePipelineStateWithFunction:metal_function error:&error]; \
        [metal_function release]; \
        if (error) { \
            printf("%s: error: load pipeline error: %s\n", __func__, [[error description] UTF8String]); \
            [metal_library release]; \
            return NULL; \
        } \
    }

    // GSPLAT_METAL_ADD_KERNEL(nd_rasterize_forward_kernel);
    // GSPLAT_METAL_ADD_KERNEL(nd_rasterize_backward_kernel);
    GSPLAT_METAL_ADD_KERNEL(rasterize_forward_kernel);
    // GSPLAT_METAL_ADD_KERNEL(rasterize_backward_kernel);
    GSPLAT_METAL_ADD_KERNEL(project_gaussians_forward_kernel);
    // GSPLAT_METAL_ADD_KERNEL(project_gaussians_backward_kernel);
    GSPLAT_METAL_ADD_KERNEL(compute_sh_forward_kernel);
    // GSPLAT_METAL_ADD_KERNEL(compute_sh_backward_kernel);
    // GSPLAT_METAL_ADD_KERNEL(compute_cov2d_bounds_kernel);
    // GSPLAT_METAL_ADD_KERNEL(map_gaussian_to_intersects_kernel);
    // GSPLAT_METAL_ADD_KERNEL(get_tile_bin_edges_kernel);

    [metal_library release];

    return ctx;
}

// TODO(achan): Where do I call this?
void free_gsplat_metal_context(MetalContext* ctx) {
    [ctx->nd_rasterize_forward_kernel_cpso release];
    [ctx->nd_rasterize_backward_kernel_cpso release];
    [ctx->rasterize_forward_kernel_cpso release];
    [ctx->rasterize_backward_kernel_cpso release];
    [ctx->project_gaussians_forward_kernel_cpso release];
    [ctx->project_gaussians_backward_kernel_cpso release];
    [ctx->compute_sh_forward_kernel_cpso release];
    [ctx->compute_sh_backward_kernel_cpso release];
    [ctx->compute_cov2d_bounds_kernel_cpso release];
    [ctx->map_gaussian_to_intersects_kernel_cpso release];
    [ctx->get_tile_bin_edges_kernel_cpso release];

    [ctx->queue release];
    [ctx->device release];
    // We do not need to release `d_queue` here as that is managed by torch.

    free(ctx);
}

MetalContext* get_global_context() {
    static MetalContext* ctx = NULL;
    if (ctx == NULL) {
        ctx = init_gsplat_metal_context();
    }
    return ctx;
}

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

std::tuple<
    torch::Tensor, // output conics
    torch::Tensor> // output radii
compute_cov2d_bounds_tensor(const int num_pts, torch::Tensor &covs2d) {
    CHECK_INPUT(covs2d);
    torch::Tensor conics = torch::zeros(
        {num_pts, covs2d.size(1)}, covs2d.options().dtype(torch::kFloat32)
    );
    torch::Tensor radii =
        torch::zeros({num_pts, 1}, covs2d.options().dtype(torch::kFloat32));

    return std::make_tuple(conics, radii);
}

torch::Tensor compute_sh_forward_tensor(
    unsigned num_points,
    unsigned degree,
    unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &coeffs
) {
    unsigned num_bases = num_sh_bases(degree);
    if (coeffs.ndimension() != 3 || coeffs.size(0) != num_points ||
        coeffs.size(1) != num_bases || coeffs.size(2) != 3) {
        AT_ERROR("coeffs must have dimensions (N, D, 3)");
    }
    torch::Tensor colors = torch::empty({num_points, 3}, coeffs.options());

    // Get a reference to the command buffer for the MPS stream
    id<MTLCommandBuffer> command_buffer = torch::mps::get_command_buffer();
    TORCH_CHECK(command_buffer, "Failed to retrieve command buffer reference");

    // Dispatch the kernel
    MetalContext* ctx = get_global_context();
    dispatch_sync(ctx->d_queue, ^(){
        // Start a compute pass
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        TORCH_CHECK(encoder, "Failed to create compute command encoder");

        // Encode the pipeline state object
        id<MTLComputePipelineState> cpso = ctx->compute_sh_forward_kernel_cpso;
        [encoder setComputePipelineState:cpso];

        // Set the tensor buffers
        [encoder setBytes:&num_points length:sizeof(num_points) atIndex:0];
        [encoder setBytes:&degree length:sizeof(degree) atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(viewdirs) offset:viewdirs.storage_offset() * viewdirs.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(coeffs) offset:coeffs.storage_offset() * coeffs.element_size() atIndex:3];
        [encoder setBuffer:getMTLBufferStorage(colors) offset:colors.storage_offset() * colors.element_size() atIndex:4];

        // Set the grid threadgroup sizes
        MTLSize grid_size = MTLSizeMake(num_points, 1, 1);
        
        NSUInteger num_threads_per_group = MIN(cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
        MTLSize thread_group_size = MTLSizeMake(num_threads_per_group, 1, 1);

        // Dispatch the compute command
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
        [encoder endEncoding];

        // Commit the work
        torch::mps::synchronize();
    });
    return colors;
}

torch::Tensor compute_sh_backward_tensor(
    unsigned num_points,
    unsigned degree,
    unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &v_colors
) {
    if (viewdirs.ndimension() != 2 || viewdirs.size(0) != num_points ||
        viewdirs.size(1) != 3) {
        AT_ERROR("viewdirs must have dimensions (N, 3)");
    }
    if (v_colors.ndimension() != 2 || v_colors.size(0) != num_points ||
        v_colors.size(1) != 3) {
        AT_ERROR("v_colors must have dimensions (N, 3)");
    }
    unsigned num_bases = num_sh_bases(degree);
    torch::Tensor v_coeffs =
        torch::zeros({num_points, num_bases, 3}, v_colors.options());
    return v_coeffs;
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_forward_tensor(
    const int num_points,
    torch::Tensor &means3d,
    torch::Tensor &scales,
    const float glob_scale,
    torch::Tensor &quats,
    torch::Tensor &viewmat,
    torch::Tensor &projmat,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const unsigned img_height,
    const unsigned img_width,
    const std::tuple<int, int, int> tile_bounds,
    const float clip_thresh
) {
    // Triangular covariance.
    torch::Tensor cov3d_d =
        torch::zeros({num_points, 6}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor xys_d =
        torch::zeros({num_points, 2}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor depths_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor radii_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kInt32));
    torch::Tensor conics_d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor num_tiles_hit_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kInt32));

    // Get a reference to the command buffer for the MPS stream
    id<MTLCommandBuffer> command_buffer = torch::mps::get_command_buffer();
    TORCH_CHECK(command_buffer, "Failed to retrieve command buffer reference");

    // Dispatch the kernel
    MetalContext* ctx = get_global_context();
    dispatch_sync(ctx->d_queue, ^(){
        // Start a compute pass
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        TORCH_CHECK(encoder, "Failed to create compute command encoder");

        float intrins[4] = {fx, fy, cx, cy};
        int32_t img_size[3] = {(int32_t)img_width, (int32_t)img_height, 1};
        int32_t tile_bounds_dim3[3] = {std::get<0>(tile_bounds), std::get<1>(tile_bounds), std::get<2>(tile_bounds)};

        // Encode the pipeline state object
        id<MTLComputePipelineState> cpso = ctx->project_gaussians_forward_kernel_cpso;
        [encoder setComputePipelineState:cpso];

        // Set the tensor buffers
        [encoder setBytes:&num_points length:sizeof(num_points) atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(means3d) offset:means3d.storage_offset() * means3d.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(scales) offset:scales.storage_offset() * scales.element_size() atIndex:2];
        [encoder setBytes:&glob_scale length:sizeof(glob_scale) atIndex:3];
        [encoder setBuffer:getMTLBufferStorage(quats) offset:quats.storage_offset() * quats.element_size() atIndex:4];
        [encoder setBuffer:getMTLBufferStorage(viewmat) offset:viewmat.storage_offset() * viewmat.element_size() atIndex:5];
        [encoder setBuffer:getMTLBufferStorage(projmat) offset:projmat.storage_offset() * projmat.element_size() atIndex:6];
        [encoder setBytes:intrins length:sizeof(intrins) atIndex:7];
        [encoder setBytes:img_size length:sizeof(img_size) atIndex:8];
        [encoder setBytes:tile_bounds_dim3 length:sizeof(tile_bounds_dim3) atIndex:9];
        [encoder setBytes:&clip_thresh length:sizeof(clip_thresh) atIndex:10];
        [encoder setBuffer:getMTLBufferStorage(cov3d_d) offset:cov3d_d.storage_offset() * cov3d_d.element_size() atIndex:11];
        [encoder setBuffer:getMTLBufferStorage(xys_d) offset:xys_d.storage_offset() * xys_d.element_size() atIndex:12];
        [encoder setBuffer:getMTLBufferStorage(depths_d) offset:depths_d.storage_offset() * depths_d.element_size() atIndex:13];
        [encoder setBuffer:getMTLBufferStorage(radii_d) offset:radii_d.storage_offset() * radii_d.element_size() atIndex:14];
        [encoder setBuffer:getMTLBufferStorage(conics_d) offset:conics_d.storage_offset() * conics_d.element_size() atIndex:15];
        [encoder setBuffer:getMTLBufferStorage(num_tiles_hit_d) offset:num_tiles_hit_d.storage_offset() * num_tiles_hit_d.element_size() atIndex:16];

        // Set the grid threadgroup sizes
        MTLSize grid_size = MTLSizeMake(num_points, 1, 1);
        NSUInteger num_threads_per_group = MIN(cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
        MTLSize thread_group_size = MTLSizeMake(num_threads_per_group, 1, 1);

        // Dispatch the compute command
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
        [encoder endEncoding];

        // Commit the work
        torch::mps::synchronize();
    });
    
    return std::make_tuple(
        cov3d_d, xys_d, depths_d, radii_d, conics_d, num_tiles_hit_d
    );
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_backward_tensor(
    const int num_points,
    torch::Tensor &means3d,
    torch::Tensor &scales,
    const float glob_scale,
    torch::Tensor &quats,
    torch::Tensor &viewmat,
    torch::Tensor &projmat,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const unsigned img_height,
    const unsigned img_width,
    torch::Tensor &cov3d,
    torch::Tensor &radii,
    torch::Tensor &conics,
    torch::Tensor &v_xy,
    torch::Tensor &v_depth,
    torch::Tensor &v_conic
) {
    // Triangular covariance.
    torch::Tensor v_cov2d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_cov3d =
        torch::zeros({num_points, 6}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_mean3d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_scale =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_quat =
        torch::zeros({num_points, 4}, means3d.options().dtype(torch::kFloat32));
    
    return std::make_tuple(v_cov2d, v_cov3d, v_mean3d, v_scale, v_quat);
}


std::tuple<torch::Tensor, torch::Tensor> map_gaussian_to_intersects_tensor(
    const int num_points,
    const int num_intersects,
    const torch::Tensor &xys,
    const torch::Tensor &depths,
    const torch::Tensor &radii,
    const torch::Tensor &cum_tiles_hit,
    const std::tuple<int, int, int> tile_bounds
) {
    CHECK_INPUT(xys);
    CHECK_INPUT(depths);
    CHECK_INPUT(radii);
    CHECK_INPUT(cum_tiles_hit);

    torch::Tensor gaussian_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt32));
    torch::Tensor isect_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt64));

    return std::make_tuple(isect_ids_unsorted, gaussian_ids_unsorted);
}

torch::Tensor get_tile_bin_edges_tensor(
    int num_intersects,
    const torch::Tensor &isect_ids_sorted
) {
    CHECK_INPUT(isect_ids_sorted);
    torch::Tensor tile_bins = torch::zeros(
        {num_intersects, 2}, isect_ids_sorted.options().dtype(torch::kInt32)
    );

    return tile_bins;
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> rasterize_forward_tensor(
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &background
) {
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(background);

    const int channels = colors.size(1);
    const int img_width = std::get<0>(img_size);
    const int img_height = std::get<1>(img_size);

    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_Ts = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_idx = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kInt32)
    );

    // Get a reference to the command buffer for the MPS stream
    id<MTLCommandBuffer> command_buffer = torch::mps::get_command_buffer();
    TORCH_CHECK(command_buffer, "Failed to retrieve command buffer reference");

    // Dispatch the kernel
    MetalContext* ctx = get_global_context();
    dispatch_sync(ctx->d_queue, ^(){
        // Start a compute pass
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        TORCH_CHECK(encoder, "Failed to create compute command encoder");

        // Encode the pipeline state object
        id<MTLComputePipelineState> cpso = ctx->rasterize_forward_kernel_cpso;
        [encoder setComputePipelineState:cpso];

        int32_t tile_bounds_dim3[3] = {std::get<0>(tile_bounds), std::get<1>(tile_bounds), std::get<2>(tile_bounds)};
        int32_t img_size_dim3[3] = {std::get<0>(img_size), std::get<1>(img_size), std::get<2>(img_size)};
        int32_t block_size_dim3[3] = {std::get<0>(block), std::get<1>(block), std::get<2>(block)};

        // Set the tensor buffers
        [encoder setBytes:tile_bounds_dim3 length:sizeof(tile_bounds_dim3) atIndex:0];
        [encoder setBytes:img_size_dim3 length:sizeof(img_size_dim3) atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(gaussian_ids_sorted) offset:gaussian_ids_sorted.storage_offset() * gaussian_ids_sorted.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(tile_bins) offset:tile_bins.storage_offset() * tile_bins.element_size() atIndex:3];
        [encoder setBuffer:getMTLBufferStorage(xys) offset:xys.storage_offset() * xys.element_size() atIndex:4];
        [encoder setBuffer:getMTLBufferStorage(conics) offset:conics.storage_offset() * conics.element_size() atIndex:5];
        [encoder setBuffer:getMTLBufferStorage(colors) offset:colors.storage_offset() * colors.element_size() atIndex:6];
        [encoder setBuffer:getMTLBufferStorage(opacities) offset:opacities.storage_offset() * opacities.element_size() atIndex:7];
        [encoder setBuffer:getMTLBufferStorage(final_Ts) offset:final_Ts.storage_offset() * final_Ts.element_size() atIndex:8];
        [encoder setBuffer:getMTLBufferStorage(final_idx) offset:final_idx.storage_offset() * final_idx.element_size() atIndex:9];
        [encoder setBuffer:getMTLBufferStorage(out_img) offset:out_img.storage_offset() * out_img.element_size() atIndex:10];
        [encoder setBuffer:getMTLBufferStorage(background) offset:background.storage_offset() * background.element_size() atIndex:11];
        [encoder setBytes:block_size_dim3 length:2*sizeof(int32_t) atIndex:12];

        // Set the grid threadgroup sizes
        MTLSize grid_size = MTLSizeMake(img_height, img_width, 1);
        // TODO(achan): we should be able to remove the 3rd dimension of `block` as it is always set to 1
        MTLSize thread_group_size = MTLSizeMake(block_size_dim3[0], block_size_dim3[1], block_size_dim3[2]);

        // Dispatch the compute command
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
        [encoder endEncoding];

        // Commit the work
        torch::mps::synchronize();
    });

    return std::make_tuple(out_img, final_Ts, final_idx);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> nd_rasterize_forward_tensor(
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &background
) {
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(background);

    const int channels = colors.size(1);
    const int img_width = std::get<0>(img_size);
    const int img_height = std::get<1>(img_size);

    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_Ts = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_idx = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kInt32)
    );

    return std::make_tuple(out_img, final_Ts, final_idx);
}


std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor  // dL_dopacity
        >
    nd_rasterize_backward_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &background,
        const torch::Tensor &final_Ts,
        const torch::Tensor &final_idx,
        const torch::Tensor &v_output, // dL_dout_color
        const torch::Tensor &v_output_alpha
    ) {
    CHECK_INPUT(xys);
    CHECK_INPUT(colors);

    const int num_points = xys.size(0);
    const int channels = colors.size(1);

    torch::Tensor v_xy = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_conic = torch::zeros({num_points, 3}, xys.options());
    torch::Tensor v_colors =
        torch::zeros({num_points, channels}, xys.options());
    torch::Tensor v_opacity = torch::zeros({num_points, 1}, xys.options());

    return std::make_tuple(v_xy, v_conic, v_colors, v_opacity);
}

std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor  // dL_dopacity
        >
    rasterize_backward_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &background,
        const torch::Tensor &final_Ts,
        const torch::Tensor &final_idx,
        const torch::Tensor &v_output, // dL_dout_color
        const torch::Tensor &v_output_alpha
    ) {
    CHECK_INPUT(xys);
    CHECK_INPUT(colors);

    const int num_points = xys.size(0);
    const int channels = colors.size(1);

    torch::Tensor v_xy = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_conic = torch::zeros({num_points, 3}, xys.options());
    torch::Tensor v_colors =
        torch::zeros({num_points, channels}, xys.options());
    torch::Tensor v_opacity = torch::zeros({num_points, 1}, xys.options());

    return std::make_tuple(v_xy, v_conic, v_colors, v_opacity);
}