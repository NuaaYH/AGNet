# AGNet: Attention Guided Network for Salient Object Detection in Optical Remote Sensing Images
Code for paper "Attention Guided Network for Salient Object Detection in Optical Remote Sensing Images", 
by Yuhan Lin, Han Sun, Ningzhong Liu, Yetong Bian, Jun Cen, and Huiyu Zhou

## Requirement
python-3.6  
pytorch-1.8.1  
torchvision  
numpy  
tqdm  
cv2

## Usage
* Clone this repo into your workstation 
  git clone https://github.com/NuaaYH/AGNet.git
* The datasets used in this paper can be download from BaiduYun: https://pan.baidu.com/s/1oDOsPZkRisCBzAtoY6SDMA  (code:hl0s)  
* Set the project format as follows:  
  ./AGNet  
  ./Dataset  
* Create the folders in AGNet as shown below:  
  ./Outputs/pred/AGNet/EORSSD(or ORSSD)/Test  
  ./Checkpoints/trained
  
 ## training  
1. Comment out line 79 of run.py like #self.net.load_state_dict......  
2. Comment out line 171 of run.py like #run.test() and ensure that the run.train() statement is executable  
3. python run.py

## testing
1. Put the model weights in ./Checkpoints/trained and ensure that line 79 of run.py is executable  
2. Comment out line 170 of run.py like #run.train() and ensure that the run.test() statement is executable   
3. python run.py

## evaluation
The evaluation code can be available at https://github.com/zyjwuyan/SOD_Evaluation_Metrics.

## Results
* The results of ours and the comparison methods in our paper can be download from BaiduYun:  
https://pan.baidu.com/s/1Vl7maJjQiQVS8VG77t5lLw  (code:arjc)
* The pre-trained model can be download from BaiduYun:  
https://pan.baidu.com/s/1AcQJjKdsk3SWPr2O6i-c0A  (code:fkci)
