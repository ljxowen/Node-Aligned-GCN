# Node-Aligned-GCN for Whole-slide Image Representation and Classification
## About
The implementation of Node-Aligned Graph Convolutional Network, the original algorithm idea is from paper:
https://openaccess.thecvf.com/content/CVPR2022/html/Guan_Node-Aligned_Graph_Convolutional_Network_for_Whole-Slide_Image_Representation_and_Classification_CVPR_2022_paper.html
## How to use
Download the dataset first. Then run `main.py.` will excute all the algorithm.
### Dataset
The dataset used for this implementation is from https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images, it provide the patch img cut from the whole-slide images (WSIs). If you are using your own dataset, you need to cut WSIs into patches first and then use this algorithm (Note: you also need write your own code to read your own data).
