# 3d_foot_reconstruction_demo

**生成ubuntu下的数据采集可执行文件（实时脚部区域分割，按s键生成对应的体素表示文件）**

设备要求
- kinect深度相机一台

环境配置
- ubuntu 16.04系统
- cmake版本大于3.1 sudo apt-get install cmake
- 安装opencv依赖 sudo apt-get install  libopencv-dev
- 安装openni依赖 sudo apt-get install libopenni-dev

代码编译
- cd到项目文件夹 cd proj_folder
- cd data_collector
- mkdir build
- cd build
- cmake ..
- make

验证代码是否可运行
- ./collector ttt ../test.oni 如可运行，则会出现图像窗口
- 按esc可退出程序

拷贝可执行文件到项目根目录
- cp collector ../../

**运行三维重建代码**

环境配置
- ubuntu 16.04系统

安装最新版meshlab(2016.12版本)
- sudo add-apt-repository ppa:zarquon42/meshlab
- sudo apt-get isntall meshlab

创建虚拟环境（建议使用anaconda）
- conda create -n 3d_foot_reconstruction python=2.7
- source activate 3d_foot_reconstruction
- pip install -r requirements.txt

运行代码
- python main.py