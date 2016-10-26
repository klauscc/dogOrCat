"""
make predictions in python.
"""
import os
import glob
import caffe
import numpy as np
from caffe.proto  import caffe_pb2
import random



def prepareNetwork(net_deploy, net_model, mean_file):
    #Read mean image
    mean_blob = caffe_pb2.BlobProto()
    with open(mean_file) as f:
        mean_blob.ParseFromString(f.read())
    mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
        (mean_blob.channels, mean_blob.height, mean_blob.width)).mean(1).mean(1)

    #Read model architecture and trained model's weights
    net = caffe.Net(net_deploy, net_model, caffe.TEST)

    #Define image transformers
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', mean_array)
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255)
    return [net, transformer]

'''
Making predicitions
'''

caffe.set_mode_gpu()
mean_file = './imagenet_mean.binaryproto'
net_deploy = './deploy.prototxt'
#net_model = '/data/tmp/klaus/projects/porn_detect/models/porn_alexnet/newdata_1000/porn_alexnet_train_iter_400.caffemodel'
net_model = '/data/tmp/klaus/dogVsCat/alexnet/dogvscat_alexnet_train_iter_10000.caffemodel'

net,transformer = prepareNetwork(net_deploy, net_model, mean_file)
predict_dir="/data/tmp/klaus/dogVsCat/test/*.jpg"
test_img_paths = [img_path for img_path in glob.glob(predict_dir)]
random.shuffle(test_img_paths)
test_img_paths = test_img_paths[0:100]
test_ids = []
preds = []

dog_count = 0
dog_error_count = 0
cat_count = 0
cat_error_count = 0

#Making predictions
for img_path in test_img_paths:
    try:
        img = caffe.io.load_image(img_path)
    except:
        print 'load image error:',img_path
        continue
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    pred_probas = out['prob']

    if 'dog' in img_path:
        dog_count = dog_count + 1
        test_ids = test_ids + [1]
        if pred_probas.argmax() != 1:
            dog_error_count = dog_error_count + 1
    else:
        test_ids = test_ids + [0]
        if pred_probas.argmax() != 0:
            cat_error_count = cat_error_count + 1
    preds = preds + [pred_probas.argmax()]
    test_imgs = test_imgs + [img_path]

    print img_path
    print pred_probas.argmax()
    print '-------'

print 'dog accuracy:', float(dog_count - dog_error_count) / dog_count, '. error:', dog_error_count, '/', dog_count
print 'cat accuracy:', float(cat_count - cat_error_count) / cat_count, '. error:', cat_error_count, '/', cat_count
print 'avg accuracy:', float(dog_count+cat_count-dog_error_count-cat_error_count) / (dog_count+cat_count)

'''
Making submission file
'''
with open("./submission_of_erotic.csv","w") as f:
    f.write("img_path,trueLabel,Predictedlabel\n")
    for i in range(len(test_ids)):
        f.write(test_imgs[i]+","+str(test_ids[i])+","+str(preds[i])+"\n")
