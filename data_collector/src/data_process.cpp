#include "data_process.h"
#include <math.h>
#include <fstream>

using namespace std;

float cx=319.5, cy=239.5;
float fx=571.2, fy=571.2;

Vec3f depth2point(int u, int v, int d)
{
    Vec3f p;
    p[0]=(u-cx)*d/fx;
    p[1]=(v-cy)*d/fy;
    p[2]=d;
    return p;
}

Mat dmap2pointCloud(const Mat& dmap, const Mat& mask)
{
    Mat res=Mat::zeros(dmap.size(),CV_32FC3);
    for (int i=0; i<dmap.rows; ++i)
    {
        for (int j=0; j<dmap.cols; ++j)
        {
            if (mask.at<uchar>(i,j)!=0)
            {
                res.at<Vec3f>(i,j)=depth2point(j,i,dmap.at<ushort>(i,j));
            }
        }
    }
    return res;
}

Mat pointCloud2norm(const Mat& pc, const Mat& dmap, Mat& mask)
{
    mask=(dmap!=0);
    Mat pc_mask=Mat::zeros(mask.size(),CV_8U);
    Mat nmap=Mat::zeros(dmap.size(),CV_32FC3);
    int offset=1;
    for (int i=dmap.rows/4; i<dmap.rows-1; ++i)
    {
        for (int j=dmap.cols/8; j<dmap.cols-dmap.cols/8-1; ++j)
        {
            uchar m0=mask.at<uchar>(i,j);
            if (m0!=0)
            {
                uchar m1=mask.at<uchar>(i,j+offset);
                uchar m2=mask.at<uchar>(i+offset,j);
                if(m1!=0 && m2!=0)
                {
                    Vec3f p0=pc.at<Vec3f>(i,j);
                    Vec3f p1=pc.at<Vec3f>(i,j+offset);
                    Vec3f p2=pc.at<Vec3f>(i+offset,j);
                    Vec3f norm_vec=computeNorm(p0,p1,p2);
                    if (norm_vec!=Vec3f(0,0,0))
                    {
                        pc_mask.at<uchar>(i,j)=255;
                        nmap.at<Vec3f>(i,j)=norm_vec;
                    }
                }
            }
        }
    }
    mask=pc_mask;
    return nmap;
}

Vec3f computeNorm(const Vec3f p0, const Vec3f p1, const Vec3f p2)
{
    Vec3f v1=p1-p0, v2=p2-p0;
    Vec3f norm;
    norm[0]=v1[1]*v2[2]-v1[2]*v2[1];
    norm[1]=v1[2]*v2[0]-v1[0]*v2[2];
    norm[2]=v1[0]*v2[1]-v1[1]*v2[0];
    float norm_length=sqrt(norm[0]*norm[0]+norm[1]*norm[1]+norm[2]*norm[2]);
    if (norm_length>0.001)
    {
        norm=norm/norm_length;
        return norm;
    }
    else
        return Vec3f(0,0,0);
}

Mat imageDenoise(const Mat& img, int type)
{
    Mat kernels=Mat::ones(5,5,CV_32F)/25.0;
    Mat res;
    filter2D(img.clone(),res,type,kernels);
    return res;
}

Mat selectPlanePoints(const Mat& nmap, const Mat& mask)
{
    Mat res=Mat::zeros(nmap.size(),CV_32S)-1;
    int counts[20][20][20]={0};
    int max_num=-1;
    int max_idx=-1;
    for (int i=0; i<nmap.rows; ++i)
    {
        for (int j=0; j<nmap.cols; ++j)
        {
            if (mask.at<uchar>(i,j)!=0)
            {
                Vec3f n=nmap.at<Vec3f>(i,j);
                int idx_0=(int)((n[0]+1)/0.1);
                int idx_1=(int)((n[1]+1)/0.1);
                int idx_2=(int)((n[2]+1)/0.1);
                counts[idx_0][idx_1][idx_2] += 1;
                int idx=idx_0*400+idx_1*20+idx_2;
                res.at<int>(i,j)=idx;
                int count_tmp=counts[idx_0][idx_1][idx_2];
                if (count_tmp>max_num)
                {
                    max_num=count_tmp;
                    max_idx=idx;
                }
                
            }
        }
    }
    return res==max_idx;
}

Mat detectPlane(const Mat& pc, const Mat& mask)
{
    // 从图像下1/4部分选均匀6个小块，每个小块分别拟合平面，如果拟合的平面足够准确，则该块内的点作为候选点
    Mat res=Mat::zeros(pc.size(),CV_8U);
    int roi_size=60;
    int start_i=120-roi_size/2, start_j=100-roi_size/2;
    for (int i=0; i<40; ++i)
    {
        Rect roi=Rect(start_j+i%8*roi_size, start_i+i/8*roi_size, roi_size, roi_size);
        Mat pc_region=pc(roi);
        Mat mask_region=mask(roi);
        Mat para=fitPlane(pc_region,mask_region);
        bool is_valid=isValidPlane(pc_region,mask_region,para);
        if (is_valid)
        {
            res(roi)=255;
        }
    }
    return res;
}

Mat fitPlane(const Mat& pc, const Mat& mask)
{
    int counts=countNonZero(mask);
    if (counts<=100)
    {
        return Mat::zeros(3,1,CV_32F);
    }
    Mat src1=Mat::zeros(counts,3,CV_32F);
    Mat src2=Mat::ones(counts,1,CV_32F);
    int idx=0;
    for (int i=0; i<mask.rows; ++i)
    {
        for (int j=0; j<mask.cols; ++j)
        {
            if (mask.at<uchar>(i,j)!=0)
            {
                Vec3f p=pc.at<Vec3f>(i,j);
                src1.at<float>(idx,0)=p[0];
                src1.at<float>(idx,1)=p[1];
                src1.at<float>(idx,2)=p[2];
                ++idx;
            }
        }
    }
    Mat para;
    solve(src1,src2,para,DECOMP_SVD);
    return para;
}

bool isValidPlane(const Mat& pc, const Mat& mask, const Mat& para)
{
    // 95%的点与平面距离在5mm以内，认为是有效平面
    int total_counts=countNonZero(mask);
    if (total_counts==0)
    {
        return false;
    }
    double a0=para.at<float>(0,0);
    double a1=para.at<float>(1,0);
    double a2=para.at<float>(2,0);
    if (a0!=0 || a1!=0 || a2!=0)
    {
        float fm=sqrt(pow(a0,2.0)+pow(a1,2.0)+pow(a2,2.0));
        int counts=0;
        for (int i=0;i<mask.rows;++i)
        {
            for (int j=0;j<mask.cols;++j)
            {
                if (mask.at<uchar>(i,j)!=0)
                {
                    Vec3f p=pc.at<Vec3f>(i,j);
                    float fz=abs(a0*p[0]+a1*p[1]+a2*p[2]-1);
                    if (fz/fm<=5)
                    {
                        ++counts;
                    }
                }
            }
        }
        if (counts*1.0/total_counts>=0.95)
        {
            return true;
        }
        else
            return false;
    }
    else
        return false;
}
void write2obj(const Mat& pc, const Mat& mask)
{
    ofstream f_out;
    f_out.open("test.obj");
    for (int i=0;i<mask.rows;++i)
    {
        for (int j=0;j<mask.cols;++j)
        {
            if (mask.at<uchar>(i,j)!=0)
            {
                f_out<<"v "<<pc.at<Vec3f>(i,j)[0]<<" "<<pc.at<Vec3f>(i,j)[1]<<" "<<pc.at<Vec3f>(i,j)[2]<<endl;
            }
        }
    }
    f_out.close();
}
Mat extractFoot(const Mat& pc, const Mat& mask,  const Mat& plane)
{
    double a0=plane.at<float>(0,0);
    double a1=plane.at<float>(1,0);
    double a2=plane.at<float>(2,0);
    Mat res=Mat::zeros(mask.size(), CV_8U);
    if (a0!=0 || a1!=0 || a2!=0)
    {
        float fm=sqrt(pow(a0,2.0)+pow(a1,2.0)+pow(a2,2.0));
        int counts=0;
        for (int i=0;i<mask.rows;++i)
        {
            for (int j=0;j<mask.cols;++j)
            {
                if (mask.at<uchar>(i,j)!=0)
                {
                    Vec3f p=pc.at<Vec3f>(i,j);
                    float fz=-(a0*p[0]+a1*p[1]+a2*p[2]-1);
                    if (fz/fm>=10 && fz/fm<=110)
                    {
                        res.at<uchar>(i,j)=255;
                    }
                }
            }
        }
    }
    return res;
}
Mat removeMaskNoise(const Mat& mask)
{
    Mat element=getStructuringElement(MORPH_RECT, Size(3,3));
    Mat m=mask.clone();
    erode(m,m,element);
    dilate(m,m,element);
    vector<vector<Point2i>> contours;
    findContours(m,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    // 找最接近中心位置并且足够大的区域，作为脚部区域
    int idx=-1;
    int  min_dis=1200;
    for (int i=0;i<contours.size();++i)
    {
        if (contours[i].size()>=200 && contours[i].size()<=1000)
        {
            float cx=0,cy=0;
            for (int j=0;j<contours[i].size();++j)
            {
                cx += contours[i][j].x;
                cy += contours[i][j].y;
            }
            cx /=contours[i].size();
            cy /=contours[i].size();
            if (abs(cx-320)+abs(cy-240)<min_dis)
            {
                idx=i;
                min_dis=abs(cx-320)+abs(cy-240);
            }
        }
    }
    if (idx<0)
    {
        return Mat::zeros(mask.size(),CV_8U);
    }
    else
    {
        vector<vector<Point2i>> contour;
        contour.push_back(contours[idx]);
        Mat res=Mat::zeros(mask.size(),CV_8U);
        drawContours(res,contour,0,255,CV_FILLED);
        res.setTo(0,mask==0);
        return res;
    }
}

string fixed_length_string(int value, int length)
{
    string num_s=to_string(value);
    string prefix=string(length-num_s.size(),'0');
    return prefix+num_s;
}
Vec3f uvd2xyz(int u, int v, int d)
{
    Vec3f res;
    res[0]=(u-cx)*d*1.0/fx;
    res[1]=-(v-cy)*d*1.0/fy;
    res[2]=-d;
    return res;
}
Vec3f xyz2uvd(float x, float y, float z)
{
    // 世界坐标系转到相机坐标系
    y=-y; z=-z;
    Vec3f pt;
    pt[0]=cx+x*fx/z;
    pt[1]=cy+y*fy/z;
    pt[2]=z;
    return pt;
}
vector<int> depth2volume(Mat dmap, int volume_size, float voxel_size)
{
    vector<int> volume(volume_size*volume_size*volume_size);
    // 计算深度图中心
    float object_cx=0, object_cy=0, object_cz=0;
    int num=0, height=dmap.rows, width=dmap.cols;
    for (int i=0;i<height;++i)
    {
        for (int j=0;j<width;++j)
        {
            int dvalue=dmap.at<ushort>(i,j);
            if (dvalue!=0)
            {
                Vec3f pts=uvd2xyz(j,i,dmap.at<ushort>(i,j));
                object_cx +=pts[0]; object_cy +=pts[1]; object_cz +=pts[2];
                num +=1;
            }
        }
    }
    object_cx /=num; object_cy /=num; object_cz /=num;
    // 以点云中心为中心的体素模型计算
    int flag=0;
    float voxel_center=float(volume_size-1.0)/2;
    for (int i=0;i<volume_size;++i)
    {
        for (int j=0;j<volume_size;++j)
        {
            for (int k=0;k<volume_size;++k)
            {
                float x=(i-voxel_center)*voxel_size+object_cx, y=-(j-voxel_center)*voxel_size+object_cy, z=-(k-voxel_center)*voxel_size+object_cz;
                Vec3f pt=xyz2uvd(x,y,z);
                if (pt[0]>=0 && pt[0]<width && pt[1]>=0 && pt[1]<height)
                {
                    int idx=i*volume_size*volume_size+j*volume_size+k;
                    int dif=pt[2]-dmap.at<ushort>(int(pt[1]),int(pt[0]));
                    if (abs(dif)<=voxel_size/2)
                    {
                        volume[idx]=1;
                        ++flag;
                    }
                    else
                    {
                        volume[idx]=0;
                    }
                }
            }
        }
    }
    cout<<flag<<endl;
    return volume;
}
void save_compressed(const vector<int>& volume, string filename)
{
    int total_num=0;
    int counts=0;
    int last_value=0;
    ofstream f_out;
    f_out.open(filename+".txt");
    for (int i=0;i<volume.size();++i)
    {
        int value=volume[i];
        if (value!=last_value)
        {
            if (counts==0)
            {
                last_value=value;
                counts=1;
            }
            else
            {
                f_out<<int(last_value)<<" "<<int(counts)<<endl;
                total_num += counts;
                last_value=value;
                counts=1;
            }
        }
        else
        {
            if (counts==255)
            {
                f_out<<int(last_value)<<" "<<int(counts)<<endl;
                total_num += counts;
                last_value=value;
                counts=1;
            }
            else
            {
                ++counts;
            }
        }
    }
    f_out<<int(last_value)<<" "<<int(counts);
    total_num += counts;
    f_out.close();
    //cout<<total_num<<endl;
}
