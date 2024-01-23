/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MVE_CAMERA_HEADER
#define MVE_CAMERA_HEADER

#include <string>

#include "mve/defines.h"

MVE_NAMESPACE_BEGIN

/**
 * Per-view camera information with various helper functions.
 * An invalid camera is indicated with focal length set to 0.0f.
 */
struct CameraInfo
{
public:
    /**
     * Creates a new camera and invalidates it (sets 'flen' to 0.0f).
     */
    CameraInfo (void);

    /**
     * Stores camera position 3-vector into array pointed to by pos.
     * This can be thought of as the camera to world translation.
     * The position is calculated with: -R^T * t.
     */
    void fill_camera_pos (float* pos) const;

    /**
     * Stores the camera translation 3-vector into array pointed to by pos.
     * This can be thought of as the world to camera translation.
     * This is identical to the translation stored in the camera.
     */
    void fill_camera_translation (float* trans) const;

    /**
     * Stores the (normalized) viewing direction in world coordinates
     * into the array pointed to by 'viewdir'.
     */
    void fill_viewing_direction (float* viewdir) const;

    /**
     * Stores world to camera 4x4 matrix in array pointed to by mat.
     */
    void fill_world_to_cam (float* mat) const;

    /**
     * Stores camera to world 4x4 matrix in array pointed to by mat.
     */
    void fill_cam_to_world (float* mat) const;

    /**
     * Stores the world to camera 3x3 rotation matrix in mat.
     * This is identical to the rotation stored in the camera.
     */
    void fill_world_to_cam_rot (float* mat) const;

    /**
     * Stores the camera to world 3x3 rotation matrix in mat.
     * This is identical to the transposed rotation stored in the camera.
     */
    void fill_cam_to_world_rot (float* mat) const;

    /**
     * Initializes 'rot' and 'trans' members using the 4x4 matrix 'mat'.
     */
    void set_transformation (float const* mat);

    /**
     * Stores the 3x3 calibration (or projection) matrix (K-matrix in
     * Hartley, Zisserman). The matrix projects a point in camera coordinates
     * to the image plane with dimensions 'width' and 'height'. The convention
     * is that the camera looks along the positive z-axis. To obtain the
     * pixel coordinates after projection, 0.5 must be subtracted from the
     * coordinates.
     *
     * If the dimensions of the image are unknown, the generic projection
     * matrix with w=1 and h=1 can be used. Image coordinates x' and y' for
     * image size w and h are then computed as follows:
     *
     * For w > h:  x' = x * w  and  y' = y * w - (w - h) / 2.
     * For h > w:  x' = x * h - (h - w) / 2  and  y' = y * h.
     */
    void fill_calibration (float* mat, float width, float height) const;

    /**
     * Stores 3x3 inverse calibration (or inverse projection) matrix.
     * The matrix projects a point (x, y, 1) from the image plane into
     * the camera coordinate system. Note that 0.5 must be added to the pixel
     * coordinates x and y before projection.
     */
    void fill_inverse_calibration (float* mat,
        float width, float height) const;

    /**
     * Stores the reprojection operator (mat, vec) from pixel coordinates
     * in this source view to the given destination view. The reprojection
     * of a pixel coordinate 'xs' in a source view with respect to depth
     * 'd' to coordinate 'xd' in a destination view is given by:
     *
     *   xd = Kd ( Rd Rs^-1 ( Ks^-1 * xs * d - ts ) + td )
     *
     * which gives rise to the reprojection operator (T,t) with
     *
     *   xd = T * xs * d + t
     *
     * Here, (T,t) is returned (mat, vec). Note that the depth in these
     * formulas represents the distance along the z-axis in the camera
     * frame, NOT the distance from the camera center.
     */
    void fill_reprojection (CameraInfo const& destination,
        float src_width, float src_height, float dst_width, float dst_height,
        float* mat, float* vec) const;

    /** Retuns the rotation in string format. */
    std::string get_rotation_string (void) const;

    /** Retuns the translation in string format. */
    std::string get_translation_string (void) const;

    /** Sets the rotation from string. */
    void set_rotation_from_string (std::string const& rot_string);

    /** Sets the translation from string. */
    void set_translation_from_string (std::string const& trans_string);

    /**
     * Stores OpenGl view transformation 4x4 matrix in array pointed to by mat.
     */
    void fill_gl_viewtrans (float* mat) const;

    /**
     * Stores OpenGl projection 4x4 matrix in array pointed to by mat.
     */
    void fill_gl_projection (float* mat, float width, float height,
        float znear, float zfar) const;

    /**
     * Prints debug information to stdout.
     */
    void debug_print (void) const;

public:
    /* Intrinsic camera parameters. */

    /** Focal length. */
    float flen;
    /** Principal point in x- and y-direction. */
    float ppoint[2];
    /** Pixel aspect ratio pixel_width / pixel_height. */
    float paspect;
    /** Image distortion parameters. */
    float dist[2];

    /* Extrinsic camera parameters. */

    /** Camera translation vector. Camera position p = -ROT^T * trans. */
    float trans[3];
    /** Camera rotation which transforms from world to cam. */
    float rot[9];
};

MVE_NAMESPACE_END

#endif /* MVE_CAMERA_HEADER */
