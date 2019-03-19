# ProjectMAP583

## Two architecture: heatmap orienated and regression orientated

##Structure of the Script:

### utils:
``utils.py``: A data preprosessing, visualisation and dataloaderfor **heatmap** model<br/>
``utils1.py``: A data preprosessing, visualisation and dataloader for **regression** model<br/>
### Models:
``EnDecoder.ipynb``: The simple encoder-decoder structure model, **heatmap** based.<br/>
``SPPE-Copy1.ipynb``: An hourglass encoder-decoder equipped with Resblock model, **heatmap** based.<br/>
``Resnet152V1.ipynb``: A transfer learning version with ResNet152, **regression** based.<br/>
``SimpleRegressor.ipynb``: A simple regressor, **regression** based.<br/>

## Data Set:
The training of our model is based on the [```Youtube Pose```](https://www.robots.ox.ac.uk/~vgg/data/pose/), which consists of 5000 labeled images. Labels are 7$\times$2 array. Those coordinates denotes the seven joints for a upper body. (Head, right wrist, left wrist, right elbow, left elbow and two shoulders.) 


### Statistics of Dataset
<p align='center'>
    <img src="Images/head.png", width="145">
    <img src="Images/left_e.png", width="150">
    <img src="Images/left_s.png", width="145">
    <img src="Images/left_w.png", width="150">
</p>
<p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    head     
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    left elbow       
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    left shoulder
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    left wrist
</p>
<p align='center'>
    <img src="Images/right_e.png", width="160">
    <img src="Images/right_s.png", width="150">
    <img src="Images/right_w.png", width="155">
</p>
<p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
right elbow
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    right shoulder
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    right wrist
</p>

***To be continued***

***标准结局***
