// 处理深度图数据，从图像中提取出脚部区域
// 主要方法：地面检测和拟合；高于地面一定阈值内的点作为脚部点
/*
    1. 算点云；2. 算法向；3. 法向去噪/滤波；
    4. 法向离散化； 5. 找离散化后的法向体素的最大值；
    6. 最大值对应的三维点进行平面拟合
*/

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

Vec3f depth2point(int u, int v, int d);
Mat dmap2pointCloud(const Mat& dmap, const Mat& mask);
Vec3f computeNorm(const Vec3f p0, const Vec3f p1, const Vec3f p2);
Mat pointCloud2norm(const Mat& pc, const Mat& dmap, Mat& mask);
Mat fitPlane(const Mat& pc, const Mat& mask);
Mat extractFoot(const Mat& pc, const Mat& mask, const Mat& plane);
Mat imageDenoise(const Mat& img, int type);
Mat selectPlanePoints(const Mat& nmap, const Mat& mask);
Mat detectPlane(const Mat& pc, const Mat& mask);
bool isValidPlane(const Mat& pc, const Mat& mask, const Mat& para);
void write2obj(const Mat& pc, const Mat& mask);
Mat removeMaskNoise(const Mat& mask);
string fixed_length_string(int value, int length);
Vec3f uvd2xyz(int u, int v, int d);
Vec3f xyz2uvd(float x, float y, float z);
vector<int> depth2volume(Mat dmap, int volume_size, float voxel_size);
void save_compressed(const vector<int>& volume, string filename);