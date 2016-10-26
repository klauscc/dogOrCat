# dogOrCat
classification Dog and Cat Use Alxenet and Caffe.

dataset: https://www.kaggle.com/c/dogs-vs-cats/data

modify the `train_val.txt` and `test.txt` to appropriate path according to your imagepath. You can generate it with `splitDataset.py`

./train.sh   #train you network. remember to modify the snapshot saving path in `alexnet/solver.prototxt` and download the imagenet model weight file in caffe model zoo
./test.sh     #test the dataset in `train_val.prototxt` in Phase TEST. You can modify the `source` to the dataset you want to test

python make_predictions.py  #predict and generate a submission file. remember modify paths.

python plot_learning_curve.py ./dogVsCat.log ./learning_curve.png  #dogVsCat.log is the training log

