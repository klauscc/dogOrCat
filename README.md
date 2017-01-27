# dogOrCat
classification Dog and Cat Use Alxenet and Caffe.

dataset: https://www.kaggle.com/c/dogs-vs-cats/data

modify the `train_val.txt` and `test.txt` to appropriate path according to your imagepath. You can generate it with `splitDataset.py`

before train the network. You must get `bvlc_alexnet.caffemodel` to dir `./alexnet`. You can get the model file in the following method:

```
cd $CAFFE_ROOT #change $CAFFE_ROOT with your caffe source code dir path or set the environment variable.
python scripts/download_model_binary.py models/bvlc_alexnet
#copy it to our dir `./alexnet`
```

`./train.sh`   #train you network. remember to modify the snapshot saving path in `alexnet/solver.prototxt` and download the imagenet model weight file in caffe model zoo
`./test.sh`     #test the dataset in `train_val.prototxt` in Phase TEST. You can modify the `source` to the dataset you want to test

`python make_predictions.py`  #predict and generate a submission file. remember modify paths.

`python plot_learning_curve.py ./dogVsCat.log ./learning_curve.png`  #dogVsCat.log is the training log

