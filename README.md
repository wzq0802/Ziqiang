 Network with edge Package
===========================

###########Environment
Linux or Windows
Python3.9
PyTorch1.7.1

All codes outline
train-main.py  //  running the code for the main training program
net.py    // contains all models of this project.
pred.py  //     running the code for the pred program
calculate-metric.py  //  caculate Dice、IoU、Precision and Recall

All codes detailed imformation
train-main.py
############# 
Description of data:
We set the input and output to 256*256 two-dimensional data.
(You can also adjust the training scale according to the needs of different research areas.)
The input data is 256*256 two-dimensional data of digital rock images 
The label data is 256*256 two-dimensional data of segmentation result (Pores are set to 0, and matrix is set to 1.)

pred.py
############# 
Since the final segmented image is a binary image, we set the threshold value to 0.5 as in typical binary images, 
where pixels with values less than 0.5 are set to 0, and those greater than or equal to 0.5 are set to 1.

calculate-metric.py
############# 
We invert the image for quantitative metric calculation to obtain results regarding the black pore regions.
