# Neural style Transfer
艺术图像数据集：WirkArt  
内容图像数据集：COCO2014  
内容图像视频数据集：DAVIS、YoutubeVOS  
## v1版：  
SANet + GNN， 利用GNN输入多帧来使用视频数据集进行训练（无效果，失败）  
## v2版：
可训练的双encoder（内容和风格的encoder分别使用两个可训练的VGG-19，然后利用一个预训练好的VGG-19作为损失网络） 
## v2版结果：  
![figure1](https://github.com/EnchanterXiao/NST_GNN/blob/master/result/result1.jpg)  
![figure2](https://github.com/EnchanterXiao/NST_GNN/blob/master/result/result2.jpg)


