#!/bin/bash

caffe train -solver ./alexnet/solver.prototxt -weights ./alexnet/bvlc_alexnet.caffemodel  2>&1| tee dogVsCat.log
