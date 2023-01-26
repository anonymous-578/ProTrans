# ProTrans
ProTrans: Prototype Aggregation for Transfer Learning

Steps to run code

1) clone this repository and change to the directory
```
git clone https://github.com/anonymous-578/ProTrans
cd ./ProTrans
```

2) set environment  
edit the prefix part of environment.yaml to your anaconda environment path  
ex) prefix: --> prefix: /home/(username)/anaconda3/envs/protrans  
Then create and activate the environment
```
conda env create -f environment.yaml
conda activate protrans
```

3) run main.py  
You can set configuration using the configuration files in configs/  
ex) configs/ProTrans/aircraft_15.yaml
```
python main.py --cfg configs/ProTrans/aircraft_15.yaml
```

Note

Currently running code on CUB-200-2011 is not avaiable because it is not downloaded from Torchvision
