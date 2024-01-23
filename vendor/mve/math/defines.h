/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MATH_DEFINES_HEADER
#define MATH_DEFINES_HEADER

#include <limits>

#define MATH_NAMESPACE_BEGIN namespace math {
#define MATH_NAMESPACE_END }

#define MATH_ALGO_NAMESPACE_BEGIN namespace algo {
#define MATH_ALGO_NAMESPACE_END }

#define MATH_GEOM_NAMESPACE_BEGIN namespace geom {
#define MATH_GEOM_NAMESPACE_END }

#define MATH_INTERNAL_NAMESPACE_BEGIN namespace internal {
#define MATH_INTERNAL_NAMESPACE_END }

/** Vector, Matrix, basic operations, etc. */
MATH_NAMESPACE_BEGIN
/** Algorithms, functors, value interpolation, etc. */
MATH_ALGO_NAMESPACE_BEGIN MATH_ALGO_NAMESPACE_END
/** Computation of geometric quantities and predicates. */
MATH_GEOM_NAMESPACE_BEGIN MATH_GEOM_NAMESPACE_END
/** Math internals. */
MATH_INTERNAL_NAMESPACE_BEGIN MATH_INTERNAL_NAMESPACE_END
MATH_NAMESPACE_END

/*
 * Some constants. Note that M_xx is not available everywhere.
 * These constants are computed with the high precision calculator at
 * http://keisan.casio.com/calculator set to 38 digits.
 */
#define MATH_PI         3.14159265358979323846264338327950288   // pi
#define MATH_PI_2       1.57079632679489661923132169163975144   // pi/2
#define MATH_PI_4       0.785398163397448309615660845819875721  // pi/4
#define MATH_1_PI       0.318309886183790671537767526745028724  // 1/pi
#define MATH_2_PI       0.636619772367581343075535053490057448  // 2/pi
#define MATH_TAU        6.283185307179586476925286766559005768  // 2*pi

#define MATH_SQRT2      1.41421356237309504880168872420969808   // sqrt(2)
#define MATH_SQRT3      1.7320508075688772935274463415058723669 // sqrt(3)
#define MATH_1_SQRT_2   0.707106781186547524400844362104849039  // 1/sqrt(2)
#define MATH_2_SQRT_PI  1.12837916709551257389615890312154517   // 2/sqrt(pi)
#define MATH_SQRT_PI    1.7724538509055160272981674833411451828 // sqrt(pi)
#define MATH_SQRT_2PI   2.506628274631000502415765284811045253  // sqrt(2*pi)

#define MATH_E          2.71828182845904523536028747135266250   // e
#define MATH_LOG2E      1.44269504088896340735992468100189214   // log_2(e)
#define MATH_LOG10E     0.434294481903251827651128918916605082  // log_10(e)
#define MATH_LOG102     0.301029995663981195213738894724493026  // log_10(2)
#define MATH_LN2        0.693147180559945309417232121458176568  // log_e(2)
#define MATH_LN10       2.30258509299404568401799145468436421   // log_e(10)

/* Infinity values. Consider using +/-FLT_MAX, or +/-DBL_MAX. */
#define MATH_POS_INF (std::numeric_limits<double>::infinity())
#define MATH_NEG_INF (-MATH_POS_INF)

/* Fast power macros. */
#define MATH_POW2(x) ((x) * (x))
#define MATH_POW3(x) (MATH_POW2(x) * (x))
#define MATH_POW4(x) (MATH_POW2(MATH_POW2(x)))
#define MATH_POW5(x) (MATH_POW4(x) * (x))
#define MATH_POW6(x) (MATH_POW3(MATH_POW2(x)))
#define MATH_POW7(x) (MATH_POW6(x) * (x))
#define MATH_POW8(x) (MATH_POW4(MATH_POW2(x)))

/* Angle conversions macros, DEG <-> RAD. */
#define MATH_RAD2DEG(x) ((x) * (180.0 / MATH_PI))
#define MATH_DEG2RAD(x) ((x) * (MATH_PI / 180.0))

/* Float and double limits and epsilon values. */
#ifndef __FLT_MIN__
#   define __FLT_MIN__ 1.17549435e-38f
#endif
#ifndef __DBL_MIN__
#   define __DBL_MIN__ 2.2250738585072014e-308
#endif
#define MATH_FLT_MIN __FLT_MIN__
#define MATH_DBL_MIN __DBL_MIN__
#define MATH_FLT_EPS (MATH_FLT_MIN * 1e8f) // approx. 1.17e-30
#define MATH_DBL_EPS (MATH_DBL_MIN * 1e58) // approx. 2.22e-250

/* Misc operations. */
#define MATH_SIGN(x) ((x) < 0 ? -1 : 1)

/* Floating-point epsilon comparisons. */
#define MATH_EPSILON_EQ(x,v,eps) (((v - eps) <= x) && (x <= (v + eps)))
#define MATH_EPSILON_LESS(x,v,eps) ((x + eps) < v)
#define MATH_FLOAT_EQ(x,v) MATH_EPSILON_EQ(x,v,MATH_FLT_EPS)
#define MATH_DOUBLE_EQ(x,v) MATH_EPSILON_EQ(x,v,MATH_DBL_EPS)
#define MATH_FLOAT_LESS(x,v) MATH_EPSILON_LESS(x,v,MATH_FLT_EPS)
#define MATH_DOUBLE_LESS(x,v) MATH_EPSILON_LESS(x,v,MATH_DBL_EPS)

/* NAN and INF checks. */
#define MATH_ISNAN(x) (x != x)
#define MATH_ISINF(x) (!MATH_ISNAN(x) && MATH_ISNAN(x - x))
#define MATH_NAN_CHECK(x) if (MATH_ISNAN(x)) { \
    std::cout << "NAN error in " << __FILE__ << ":" << __LINE__ << std::endl; }

/* Max std::size_t value. */
#define MATH_MAX_SIZE_T ((std::size_t)-1)
#define MATH_MAX_UINT ((unsigned int)-1)

#endif /* MATH_DEFINES_HEADER */
