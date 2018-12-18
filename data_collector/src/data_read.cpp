#include "data_read.h"
#include "data_process.h"

bool checkOpenNIError(XnStatus rc, string status)
{
    if(rc != XN_STATUS_OK){
        cerr<<status<<" Error: "<<xnGetStatusString(rc)<<endl;
        return false;
    }
    else
        return true;
}

void read_data(string folder_name, string dataName, bool is_online)
{
    
    int frameCnt=0;
    float zoomFactor=1;
    XnStatus rc=XN_STATUS_OK;

    xn::Context ctx;
    rc=ctx.Init();
    if (!checkOpenNIError(rc, "init status"))
    {
        system("pause");
        return;
    }

    xn::Player plyr;
    if (!is_online)
    {
        const char* filename=dataName.c_str();
        rc=ctx.OpenFileRecording(filename, plyr);
    }
    int frameOffset=1;
    plyr.SeekToFrame("depth", frameOffset, XN_PLAYER_SEEK_SET);
    plyr.SetRepeat(false);
    if (!checkOpenNIError(rc, "ctx.openFileRecording"))
    {
        system("pause");
        return;
    }

    xn::DepthGenerator dg;
    rc=dg.Create(ctx);
    if (!checkOpenNIError(rc, "create dg"))
    {
        system("pause");
        return;
    }
    xn::DepthMetaData depthMD;
    xn::ImageMetaData colorMD;

    XnMapOutputMode mapMode;
    dg.GetMapOutputMode(mapMode);

    rc=ctx.StartGeneratingAll();
    if (!checkOpenNIError(rc, "ctx.startGeneratingAll"))
    {
        system("pause");
        return;
    }

    char key=0;
    int counts=1;
    while (1)
    {
        frameCnt++;
        int fid=depthMD.FrameID();
        if (frameCnt>fid+1)
        {
            waitKey(0); //pause at last frame
            break; //exist after pressing any keys
        }
        ctx.WaitAndUpdateAll();
        
        dg.GetMetaData(depthMD);
        
        Mat dmat(depthMD.FullYRes(), depthMD.FullXRes(), CV_16UC1, (void*)depthMD.Data());
        //Mat cmat(colorMD.FullYRes(), colorMD.FullXRes(), CV_8UC3, (void*)colorMD.Data());
        //flip(cmat, cmat, 1);
        Mat mask=(dmat!=0);
        //dmat=imageDenoise(dmat, CV_16U);
        //dmat.setTo(0,mask==0);
        double m1,m2;
        minMaxLoc(dmat, &m1, &m2, 0,0,dmat!=0);
        printf("min_d: %f ; max_d: %f\n", m1, m2);
        flip(dmat,dmat,1);
        Mat dshow;
        dmat.convertTo(dshow, CV_8U, 255.0/2000);
        //dshow=255-dshow;
        dshow.setTo(0, dmat==0);
        Mat pc=dmap2pointCloud(dmat, mask);
        //Mat nmap=pointCloud2norm(pc, dmat, mask);
        //write2obj(pc, mask);
        //nmap=imageDenoise(nmap, CV_32F);
        Mat planeRegion=detectPlane(pc,mask);
        planeRegion.setTo(0,mask==0);
        Mat plane_para=fitPlane(pc, planeRegion);
        //imshow("color image", cmat);
        imshow("raw depth image", dshow);
        //imshow("normal map",abs(nmap));
        //imshow("mask", mask);
        //imshow("plane candidate",planeRegion);
        Mat footRegionRaw=extractFoot(pc, mask, plane_para);
        Mat footRegion=removeMaskNoise(footRegionRaw);
        imshow("foot region", dshow.setTo(0,footRegion==0));
        if (frameCnt%10 == 0)
            key=waitKey(20);
        else
            key=waitKey(20);

        //save image
        if (key=='s')
        {
            string filename=folder_name+"/"+fixed_length_string(counts,3);
            dmat.setTo(0,footRegion==0);
            Mat dshow=dmat.clone();
            dshow.setTo(400,dshow==0);
            dshow=dshow-400;
            dshow.convertTo(dshow,CV_8U,255.0/500);
            imwrite(filename+".png",dshow);
            vector<int> volume=depth2volume(dmat,64,6.0);
            save_compressed(volume,filename);
            ++counts;
        }
        if (key==27)
            break;
    }
}
