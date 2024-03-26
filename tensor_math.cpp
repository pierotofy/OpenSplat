#include "tensor_math.hpp"

using namespace torch::indexing;

torch::Tensor quatToRotMat(const torch::Tensor &quat){
    auto u = torch::unbind(torch::nn::functional::normalize(quat, torch::nn::functional::NormalizeFuncOptions().dim(-1)), -1);
    torch::Tensor w = u[0];
    torch::Tensor x = u[1];
    torch::Tensor y = u[2];
    torch::Tensor z = u[3];
    return torch::stack({
        torch::stack({
            1.0 - 2.0 * (y.pow(2) + z.pow(2)),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y)
        }, -1),
        torch::stack({
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x.pow(2) + z.pow(2)),
            2.0 * (y * z - w * x)
        }, -1),
        torch::stack({
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x.pow(2) + y.pow(2))
        }, -1)
    }, -2);
}

std::tuple<torch::Tensor, torch::Tensor, float> autoScaleAndCenterPoses(const torch::Tensor &poses){
    // Center at mean
    torch::Tensor origins = poses.index({"...", Slice(None, 3), 3});
    torch::Tensor center = torch::mean(origins, 0);
    origins -= center;

    // Scale
    float f = 1.0f / torch::max(torch::abs(origins)).item<float>();
    origins *= f;
    
    torch::Tensor transformedPoses = poses.clone();
    transformedPoses.index_put_({"...", Slice(None, 3), 3}, origins);

    return std::make_tuple(transformedPoses, center, f);
}


torch::Tensor rotationMatrix(const torch::Tensor &a, const torch::Tensor &b){
    // Rotation matrix that rotates vector a to vector b
    torch::Tensor a1 = a / a.norm();
    torch::Tensor b1 = b / b.norm();
    torch::Tensor v = torch::linalg_cross(a1, b1);
    torch::Tensor c = torch::dot(a1, b1);
    const float EPS = 1e-8;
    if (c.item<float>() < -1 + EPS){
        torch::Tensor eps = (torch::rand(3) - 0.5f) * 0.01f;
        return rotationMatrix(a1 + eps, b1);
    }
    torch::Tensor s = v.norm();
    torch::Tensor skew = torch::zeros({3, 3}, torch::kFloat32);
    skew[0][1] = -v[2];
    skew[0][2] = v[1];
    skew[1][0] = v[2];
    skew[1][2] = -v[0];
    skew[2][0] = -v[1];
    skew[2][1] = v[0];

    return torch::eye(3) + skew + torch::matmul(skew, skew * ((1 - c) / (s.pow(2) + EPS)));
}

torch::Tensor rodriguesToRotation(const torch::Tensor &rodrigues){
    float theta = torch::linalg::vector_norm(rodrigues, 2, { -1 }, true, torch::kFloat32).item<float>();
    if (theta < FLOAT_EPS){
        return torch::eye(3, torch::kFloat32);
    }
    torch::Tensor r = rodrigues / theta;
    torch::Tensor ident = torch::eye(3, torch::kFloat32);
    float a = r[0].item<float>();
    float b = r[1].item<float>();
    float c = r[2].item<float>();
    torch::Tensor rrT = torch::tensor({
        {a * a, a * b, a * c},
        {b * a, b * b, b * c},
        {c * a, c * b, c * c}
    }, torch::kFloat32);
    torch::Tensor rCross = torch::tensor({
        {0.0f, -c, b},
        {c, 0.0f, -a},
        {-b, a, 0.0f}
    }, torch::kFloat32);
    float cosTheta = std::cos(theta);

    return cosTheta * ident + (1 - cosTheta) * rrT + std::sin(theta) * rCross;
}