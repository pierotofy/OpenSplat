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

std::tuple<torch::Tensor, torch::Tensor> autoOrientAndCenterPoses(const torch::Tensor &poses){
    // Center at mean and orient up
    torch::Tensor origins = poses.index({"...", Slice(None, 3), 3});
    torch::Tensor translation = torch::mean(origins, 0);
    torch::Tensor up = torch::mean(poses.index({Slice(), Slice(None, 3), 1}), 0);
    up = up / up.norm();
    
    torch::Tensor rotation = rotationMatrix(up, torch::tensor({0, 0, 1}, torch::kFloat32));
    torch::Tensor transform = torch::cat({rotation, torch::matmul(rotation, -translation.index({"...", None}))}, -1);
    torch::Tensor orientedPoses = torch::matmul(transform, poses);
    return std::make_tuple(orientedPoses, transform);
}


torch::Tensor rotationMatrix(const torch::Tensor &a, const torch::Tensor &b){
    // Rotation matrix that rotates vector a to vector b
    torch::Tensor a1 = a / a.norm();
    torch::Tensor b1 = b / b.norm();
    torch::Tensor v = torch::cross(a1, b1);
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