// 读取kinect中的深度图数据，并对数据进行处理

#include <opencv2/opencv.hpp>
#include <XnCppWrapper.h>
#include <string>

using namespace cv;
using namespace std;

bool checkOpenNIError(XnStatus rc, string status);
void read_data(string folder_name, string dataName, bool is_online);
