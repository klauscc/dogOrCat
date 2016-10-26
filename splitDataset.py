import numpy as np
import glob
import os

image_data = [img for img in glob.glob("train/*")]

f_tv = open('train_val.txt','w')
f_test = open('test.txt','w')

os.system("mkdir -p split/train_val")
os.system("mkdir -p split/test")

for in_idx, img_path in enumerate(image_data):
    if 'dog' in img_path:
        label=1
    else:
        label=0

    if in_idx % 5 ==0:
        f_test.write("%s %d\n" %(img_path, label))
        #os.system("cp "+img_path+" split/test")
    else:
        f_tv.write("%s %d\n" %(img_path, label))
        #os.system("cp "+img_path+" split/train_val")

f_tv.close()
f_test.close()
