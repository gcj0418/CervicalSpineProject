# DataSet
ISBI 2015 and ISBI 2023. The dataset needs to be downloaded by yourself.
1. ISBI 2015 Challenge Dataset: It is a widely used benchmark dataset in the field of cephalometric landmark detection. The dataset comprises 400 cephalometric images, with 150 images used for training, 150 for Test 1, and the remaining images for Test 2. Each image is annotated by two experienced doctors, marking 19 landmarks, and the average of the two annotations is taken as the ground truth.
2. ISBI 2023 Challenge Dataset: It is a new cephalometric landmark detection dataset collected from 7 different imaging devices. Currently, only the training set of this dataset has been released, containing 700 images. According to reference [32], we randomly selected 500 images from it as unlabeled target domain data, and the remaining 200 images were used to evaluate the performance of the models, with the results averaged over 10 repeated experiments. The dataset provides 29 landmarks, but only the same 19 landmarks as in the ISBI 2015 dataset are used for comparison.

# Environment
1. python 3.7.0

2. pytorch 11.3.1

# Date preprocessing
Before running the training and testing scripts, data preprocessing is required.
1. ISBI 2015     
python ./ data_processing2015.py

2. ISBI 2023       
python ./ data_processing2023.py

# Train  
1. ISBI 2015     
python ./train_ceph2015.py 

2. ISBI 2023     
python ./train_reg2023.py

# Test
1. ISBI 2015     
python ./test_ceph2015.py

2. ISBI 2023     
python ./test_reg2023.py 
