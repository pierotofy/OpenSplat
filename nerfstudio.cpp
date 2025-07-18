#include <filesystem>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include "nerfstudio.hpp"
#include "point_io.hpp"
#include "cv_utils.hpp"
#include "tensor_math.hpp"

namespace fs = std::filesystem;

using json = nlohmann::json;
using namespace torch::indexing;

namespace ns{

void to_json(json &j, const Frame &f){
    j = json{ {"file_path", f.filePath }, 
                {"w", f.width }, 
                {"h", f.height },
                {"fl_x", f.fx },
                {"fl_y", f.fy },
                {"cx", f.cx },
                {"cy", f.cy },
                {"k1", f.k1 },
                {"k2", f.k2 },
                {"p1", f.p1 },
                {"p2", f.p2 },
                {"k3", f.k3 },
                {"transform_matrix", f.transformMatrix },
                
            };
}

void from_json(const json& j, Frame &f){
    j.at("file_path").get_to(f.filePath);
    j.at("transform_matrix").get_to(f.transformMatrix);
    if (j.contains("w")) j.at("w").get_to(f.width);
    if (j.contains("h")) j.at("h").get_to(f.height);
    if (j.contains("fl_x")) j.at("fl_x").get_to(f.fx);
    if (j.contains("fl_y")) j.at("fl_y").get_to(f.fy);
    if (j.contains("cx")) j.at("cx").get_to(f.cx);
    if (j.contains("cy")) j.at("cy").get_to(f.cy);
    if (j.contains("k1")) j.at("k1").get_to(f.k1);
    if (j.contains("k2")) j.at("k2").get_to(f.k2);
    if (j.contains("p1")) j.at("p1").get_to(f.p1);
    if (j.contains("p2")) j.at("p2").get_to(f.p2);
    if (j.contains("k3")) j.at("k3").get_to(f.k3);
    
}

void to_json(json &j, const Transforms &t){
    j = json{ {"camera_model", t.cameraModel }, 
                {"frames", t.frames },
                {"ply_file_path", t.plyFilePath },
            };
}

void from_json(const json& j, Transforms &t){
    j.at("camera_model").get_to(t.cameraModel);
    j.at("frames").get_to(t.frames);
    if (j.contains("ply_file_path")) j.at("ply_file_path").get_to(t.plyFilePath);

    // Globals
    int width = 0;
    int height = 0;
    float fx = 0;
    float fy = 0;
    float cx = 0;
    float cy = 0;
    float k1 = 0;
    float k2 = 0;
    float k3 = 0;
    float p1 = 0;
    float p2 = 0;

    if (j.contains("w")) j.at("w").get_to(width);
    if (j.contains("h")) j.at("h").get_to(height);
    if (j.contains("fl_x")) j.at("fl_x").get_to(fx);
    if (j.contains("fl_y")) j.at("fl_y").get_to(fy);
    if (j.contains("cx")) j.at("cx").get_to(cx);
    if (j.contains("cy")) j.at("cy").get_to(cy);
    if (j.contains("k1")) j.at("k1").get_to(k1);
    if (j.contains("k2")) j.at("k2").get_to(k2);
    if (j.contains("p1")) j.at("p1").get_to(p1);
    if (j.contains("p2")) j.at("p2").get_to(p2);
    if (j.contains("k3")) j.at("k3").get_to(k3);

    // Assign per-frame intrinsics if missing
    for (Frame &f : t.frames){
        if (!f.width && width) f.width = width;
        if (!f.height && height) f.height = height;
        if (!f.fx && fx) f.fx = fx;
        if (!f.fy && fy) f.fy = fy;
        if (!f.cx && cx) f.cx = cx;
        if (!f.cy && cy) f.cy = cy;
        if (!f.k1 && k1) f.k1 = k1;
        if (!f.k2 && k2) f.k2 = k2;
        if (!f.p1 && p1) f.p1 = p1;
        if (!f.p2 && p2) f.p2 = p2;
        if (!f.k3 && k3) f.k3 = k3;
    }

    std::sort(t.frames.begin(), t.frames.end(), 
        [](Frame const &a, Frame const &b) {
            return a.filePath < b.filePath; 
        });
}    

Transforms readTransforms(const std::string &filename){
    std::ifstream f(filename);
	if ( !f.is_open() )
		throw std::runtime_error( std::string("Failed to open nerf transforms file ") + filename );
    json data = json::parse(f);
    f.close();
    return data.template get<Transforms>();
}

torch::Tensor posesFromTransforms(const Transforms &t)
{
    torch::Tensor poses = torch::zeros({static_cast<long int>(t.frames.size()), 4, 4}, torch::kFloat32);
	
    for (size_t c = 0; c < t.frames.size(); c++)
	{
		auto& Camera = t.frames[c];
		Camera.CopyTransformToPoseInArray( poses, static_cast<int>(c) );
    }
    return poses;
}

	


InputData inputDataFromNerfStudio(const std::string &projectRoot,bool CenterAndNormalisePoints,bool AddCameras)
{
    InputData ret;
    fs::path nsRoot(projectRoot);
    fs::path transformsPath = nsRoot / "transforms.json";
    if (!fs::exists(transformsPath)) 
		throw std::runtime_error(transformsPath.string() + " does not exist");

    Transforms t = readTransforms(transformsPath.string());
    if (t.plyFilePath.empty()) 
		throw std::runtime_error("ply_file_path is empty");
    auto pSet = readPointSet((nsRoot / t.plyFilePath).string());

    torch::Tensor unorientedPoses = posesFromTransforms(t);

	torch::Tensor poses;
	float3 center;
	float normalisingScale = 1;
	if ( CenterAndNormalisePoints )
	{
		auto r = autoScaleAndCenterPoses(unorientedPoses);
		torch::Tensor poses = std::get<0>(r);
		center = std::get<1>(r);
		normalisingScale = std::get<2>(r);
	}
	else
	{
		//	merge together
		torch::Tensor origins = unorientedPoses.index({"...", Slice(None, 3), 3});
		torch::Tensor transformedPoses = unorientedPoses.clone();
		transformedPoses.index_put_({"...", Slice(None, 3), 3}, origins);
		poses = transformedPoses;
	}

	if ( AddCameras )
	{
		for (size_t i = 0; i < t.frames.size(); i++){
			Frame f = t.frames[i];
			
			auto Intrinsics = f.GetIntrinsics();
			
			ret.cameras.emplace_back(Camera(Intrinsics,  
											poses[i], (nsRoot / f.filePath).string()));
		}
	}
	
    torch::Tensor points = pSet->pointsTensor().clone();
    
    ret.points.xyz = points;
    ret.points.rgb = pSet->colorsTensor().clone();

	ret.TransformPoints( center, normalisingScale );
	

    return ret;
}

}



CameraIntrinsics ns::Frame::GetIntrinsics() const
{
	CameraIntrinsics intrinsics;
	intrinsics.imageWidth = width;
	intrinsics.imageHeight = height;
	intrinsics.fx = static_cast<float>(fx);
	intrinsics.fy = static_cast<float>(fy); 
	intrinsics.cx = static_cast<float>(cx);
	intrinsics.cy = static_cast<float>(cy); 
	intrinsics.k1 = static_cast<float>(k1);
	intrinsics.k2 = static_cast<float>(k2);
	intrinsics.k3 = static_cast<float>(k3); 
	intrinsics.p1 = static_cast<float>(p1); 
	intrinsics.p1 = static_cast<float>(p2);
	return intrinsics;
}

void ns::Frame::CopyTransformToPoseInArray(torch::Tensor& Pose4x4s,int PoseIndex) const
{
	//	validate input data
	int RowCount = 4;
	int ColumnCount = 4;
	
	if ( transformMatrix.size() != RowCount )
	{
		std::stringstream Error;
		Error << "TransformMatrix in camera/frame " << this->filePath << " has " << transformMatrix.size() << " rows, expected" << RowCount; 
		throw std::runtime_error(Error.str());
	}
	
	for (size_t y=0;	y<RowCount; y++)
	{
		auto& Row = transformMatrix[y];
		if ( Row.size() != ColumnCount )
		{
			std::stringstream Error;
			Error << "TransformMatrix row " << y << " in camera/frame " << this->filePath << " has " << Row.size() << " columns, expected" << ColumnCount; 
			throw std::runtime_error(Error.str());
		}
		for (size_t x=0;	x<ColumnCount;	x++)
		{
			Pose4x4s[PoseIndex][y][x] = Row[x];
		}
	}
	
}
