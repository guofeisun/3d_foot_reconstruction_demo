// ��ȡkinect�е����ͼ���ݣ��������ݽ��д���

#include <opencv2/opencv.hpp>
#include <XnCppWrapper.h>
#include <string>

using namespace cv;
using namespace std;

bool checkOpenNIError(XnStatus rc, string status);
void read_data(string folder_name, string dataName, bool is_online);
