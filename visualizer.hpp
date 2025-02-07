#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <memory>

#include <pangolin/display/view.h>
#include <pangolin/display/widgets.h>
#include <pangolin/plot/plotter.h>
#include <pangolin/var/var.h>
#include <torch/torch.h>

class Visualizer {
 public:
  Visualizer() = default;
  ~Visualizer() = default;

  bool Initialize(int iter_num = -1);

  void SetInitialGaussianNum(int num);

  void SetLoss(int step, float loss);

  void SetGaussians(const torch::Tensor& means,
                    const torch::Tensor& covariances,
                    const torch::Tensor& colors,
                    const torch::Tensor& opacities);

  void SetImage(const torch::Tensor& rendered_img, const torch::Tensor& gt_img);

  void Draw();

 private:
  bool DrawGaussians();

  bool DrawImage();

 private:
  std::unique_ptr<pangolin::OpenGlRenderState> cam_state_;
  std::unique_ptr<pangolin::View> point_cloud_viewer_;
  std::unique_ptr<pangolin::View> render_viewer_;
  std::unique_ptr<pangolin::Plotter> loss_viewer_;
  pangolin::DataLog loss_log_;
  std::unique_ptr<pangolin::Panel> panel_viewer_;
  std::unique_ptr<pangolin::Var<int>> step_;
  std::unique_ptr<pangolin::Var<int>> init_gaussian_num_;
  std::unique_ptr<pangolin::Var<int>> gaussian_num_;
  std::unique_ptr<pangolin::Var<float>> loss_;
  std::unique_ptr<pangolin::Var<bool>> pause_button_;

  torch::Tensor means_;
  torch::Tensor covariances_;
  torch::Tensor colors_;
  torch::Tensor opacities_;

  torch::Tensor rendered_img_;
  torch::Tensor gt_img_;
};

#endif
