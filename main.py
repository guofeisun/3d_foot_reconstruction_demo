from online_demo import demo
import os
import time

if __name__ == '__main__':
    d = demo()
    t = int(round(time.time()*1000))
    now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(t/1000))
    folder_name = 'data/' + now
    os.mkdir(folder_name)

    os.system('./collector ' + folder_name + ' test.oni')

    d.run(folder_name)