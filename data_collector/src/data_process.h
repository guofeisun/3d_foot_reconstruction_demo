// �������ͼ���ݣ���ͼ������ȡ���Ų�����
// ��Ҫ���������������ϣ����ڵ���һ����ֵ�ڵĵ���Ϊ�Ų���
/*
    1. ����ƣ�2. �㷨��3. ����ȥ��/�˲���
    4. ������ɢ���� 5. ����ɢ����ķ������ص����ֵ��
    6. ���ֵ��Ӧ����ά�����ƽ�����
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