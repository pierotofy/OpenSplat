#import "bindings.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

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