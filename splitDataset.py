import numpy as np
import glob
import os

database_dir="./data"

image_data = [img for img in glob.glob(database_dir+"/train/*")]

f_tv = open('train_val.txt','w')
f_test = open('test.txt','w')


for in_idx, img_path in enumerate(image_data):
    if 'dog' in img_path:
        label=1
    else:
        label=0

    if in_idx % 5 ==0:
        f_test.write("%s %d\n" %(img_path, label))
    else:
        f_tv.write("%s %d\n" %(img_path, label))

f_tv.close()
f_test.close()
