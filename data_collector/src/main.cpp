#include "data_read.h"

int main(int argc, char *argv[])
{
    if (argc<2)
    {
        cout<<"please add args of saving folder name!\n";
        return 0;
    }
    else
    {
        string filename;
        bool is_online;
        if (argc==2)
        {
            filename="";
            is_online=true;
        }
        else
        {
            filename=argv[2];
            is_online=false;
        }
        string folder_name = argv[1];
        read_data(folder_name, filename, is_online);
        return 0;
    }
}
