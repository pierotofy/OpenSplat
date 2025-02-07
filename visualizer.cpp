#include "visualizer.hpp"

#include <algorithm>
#include <chrono>
#include <thread>

#include <pangolin/display/display.h>

bool Visualizer::Initialize(int iter_num) {
  pangolin::CreateWindowAndBind("OpenSplat", 1200, 1000);
  glEnable(GL_DEPTH_TEST);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  cam_state_ = std::make_unique<pangolin::OpenGlRenderState>(
      pangolin::ProjectionMatrix(1200, 1000, 420, 420, 600, 500, 0.1f, 1000),
      pangolin::ModelViewLookAt(-1, 1, -1, 0, 0, 0, pangolin::AxisNegY));

  point_cloud_viewer_ = std::make_unique<pangolin::View>();
  point_cloud_viewer_->SetBounds(1 / 4.0f, 1.0f, 0.0f, 1 / 2.0f, true);
  point_cloud_viewer_->SetHandler(new pangolin::Handler3D(*cam_state_));
  pangolin::DisplayBase().AddDisplay(*point_cloud_viewer_);

  render_viewer_ = std::make_unique<pangolin::View>();
  render_viewer_->SetBounds(1 / 4.0f, 1.0f, 1 / 2.0f, 1.0f, true);
  pangolin::DisplayBase().AddDisplay(*render_viewer_);

  loss_log_.SetLabels({"loss"});
  float plotter_range_x = iter_num > 0 ? iter_num : 2000.0f;
  float plotter_range_y = 0.3;
  loss_viewer_ = std::make_unique<pangolin::Plotter>(
      &loss_log_, 0.0f, plotter_range_x, 0.0f, plotter_range_y, 1.f, 0.01f);
  loss_viewer_->SetBounds(0.0f, 1 / 4.0f, 0.0f, 2 / 3.0f, true);
  loss_viewer_->Track("$i");
  pangolin::DisplayBase().AddDisplay(*loss_viewer_);

  panel_viewer_ = std::make_unique<pangolin::Panel>("panel");
  panel_viewer_->SetBounds(0.0f, 1 / 4.0f, 2 / 3.0f, 1.0f, true);
  pangolin::DisplayBase().AddDisplay(*panel_viewer_);

  step_ = std::make_unique<pangolin::Var<int>>("panel.step", 0);
  init_gaussian_num_ =
      std::make_unique<pangolin::Var<int>>("panel.init gaussian num", 0);
  gaussian_num_ = std::make_unique<pangolin::Var<int>>("panel.gaussian num", 0);
  loss_ = std::make_unique<pangolin::Var<float>>("panel.loss", 0.0f);
  pause_button_ =
      std::make_unique<pangolin::Var<bool>>("panel.Start/Pause", false, false);

  return true;
}

void Visualizer::SetLoss(int step, float loss) {
  loss_log_.Log(loss);

  if (loss_viewer_) {
    pangolin::XYRangef& range = loss_viewer_->GetView();
    if (loss > range.y.max) {
      range.y.max = loss;
    }
  }

  if (loss_) {
    *loss_ = loss;
  }
  if (step_) {
    *step_ = step;
  }
}

void Visualizer::SetInitialGaussianNum(int num) {
  if (init_gaussian_num_) {
    *init_gaussian_num_ = num;
  }
}

void Visualizer::SetGaussians(const torch::Tensor& means,
                              const torch::Tensor& covariances,
                              const torch::Tensor& colors,
                              const torch::Tensor& opacities) {
  means_ = means.cpu();
  covariances_ = covariances.cpu();
  colors_ = colors.cpu();
  opacities_ = opacities.cpu();

  if (gaussian_num_) {
    *gaussian_num_ = means_.size(0);
  }
}

void Visualizer::SetImage(const torch::Tensor& rendered_img,
                          const torch::Tensor& gt_img) {
  rendered_img_ = (rendered_img.cpu() * 255).to(torch::kUInt8);
  gt_img_ = (gt_img.cpu() * 255).to(torch::kUInt8);
}

void Visualizer::Draw() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  DrawGaussians();
  DrawImage();

  pangolin::FinishFrame();

  while (*pause_button_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    pangolin::WindowInterface* window = pangolin::GetBoundWindow();
    if (window) {
      window->ProcessEvents();
    } else {
      break;
    }
  }
}

bool Visualizer::DrawGaussians() {
  if (!point_cloud_viewer_) return false;

  static const double c0 = 0.28209479177387814;
  auto sh2rgb = [](float sh) {
    return static_cast<float>(std::max(std::min(sh * c0 + 0.5, 1.0), 0.0));
  };

  point_cloud_viewer_->Activate(*cam_state_);
  glColor3f(1.0, 1.0, 1.0);

  int gaussian_num = means_.size(0);
  auto mean_accessor = means_.accessor<float, 2>();
  auto color_accessor = colors_.accessor<float, 2>();

  glBegin(GL_POINTS);
  for (int i = 0; i < gaussian_num; ++i) {
    glColor3f(sh2rgb(color_accessor[i][0]), sh2rgb(color_accessor[i][1]),
              sh2rgb(color_accessor[i][2]));
    glVertex3f(mean_accessor[i][0], mean_accessor[i][1], mean_accessor[i][2]);
  }
  glEnd();

  return true;
}

bool Visualizer::DrawImage() {
  if (!render_viewer_) return false;

  torch::Tensor concatenated_img;
  concatenated_img = torch::cat({rendered_img_, gt_img_}, 0);

  const int width = concatenated_img.size(1);
  const int height = concatenated_img.size(0);
  pangolin::GlTexture imageTexture(width, height, GL_RGB, false, 0, GL_RGB,
                                   GL_UNSIGNED_BYTE);
  unsigned char* data = concatenated_img.data_ptr<unsigned char>();
  imageTexture.Upload(data, GL_RGB, GL_UNSIGNED_BYTE);

  render_viewer_->SetAspect(static_cast<float>(width) / height);
  render_viewer_->Activate();
  glColor3f(1.0, 1.0, 1.0);
  imageTexture.RenderToViewport(true);

  return true;
}
