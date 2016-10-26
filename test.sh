#!/bin/bash

caffe test -model ./alexnet/train_val.prototxt -weights /data/tmp/klaus/dogVsCat/alexnet/dogvscat_alexnet_train_iter_10000.caffemodel  2>&1| tee dogVsCat_test.log
