# yolo3-keras-tensorflow
A tweak to https://github.com/qqwweee/keras-yolo3 to store images and json attributes


Example Data

{"boxes": "Found 8 boxes for img", "class7": "chair", "score7": "0.30609933", "class6": "chair", "score6": "0.36806455", "class5": "chair", "score5": "0.39490885", "class4": "chair", "score4": "0.5172797", "class3": "chair", "score3": "0.59243447", "class2": "chair", "score2": "0.943404", "class1": "chair", "score1": "0.99435705", "class0": "person", "score0": "0.9988756", "host": "HW13125.local", "end": "1527380495.2494621", "te": "1.010077953338623", "battery": 100, "systemtime": "05/26/2018 20:21:35", "cpu": 18.2, "diskusage": "140808.3 MB", "memory": 63.8}

HCC Article

https://community.hortonworks.com/articles/193868/integrating-keras-tensorflow-yolov3-into-apache-ni.html

This is a fork.

Follow this process:

git clone https://github.com/qqwweee/keras-yolo3 
 
wget https://pjreddie.com/media/files/yolov3.weights 
 
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5 
