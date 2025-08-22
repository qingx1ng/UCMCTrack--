# UCMCTrack-推理优化版本

本项目以UCMCTrackd作为baseline，进行推理优化任务。仓库中主要做的优化改进在tracker/ucmc.py和tracker/kalman.py中，tracker目录中的kalmanbase.py和ucmcbase.py为作者源代码仓库的文件。若需进行对比实验可把两个base代码和优化后的代码进行替换做对比。本仓库配好环境可一键跑通，在不失大精度情况下，速率得到了一定的提升。

## 实验结果展示

### baseline（加cmc）

baseline为用kalmanbase.py和ucmcbase.py脚本替代tracker/kalman.py和tracker/ucmc.py

| ***\*序列\**** | ***\*帧数\**** | ***\*耗时 Time cost (s)\**** | ***\*FPS (Frames/s)\**** | ***\*HOTA\**** | ***\*MOTA\**** | ***\*IDF1\**** | ***\*备注\**** |
| -------------- | -------------- | ---------------------------- | ------------------------ | -------------- | -------------- | -------------- | -------------- |
| MOT17-02-SDP   | 600            | 12.66                        | 47.4                     | 61.68          | 72.57          | 72.90          | baseline       |
| MOT17-04-SDP   | 1050           | 41.19                        | 25.5                     | 84.54          | 94.91          | 94.63          | baseline       |
| MOT17-05-SDP   | 837            | 1.72                         | 486.6                    | 66.55          | 82.44          | 79.77          | baseline       |
| MOT17-09-SDP   | 525            | 1.31                         | 400.8                    | 74.71          | 91.25          | 88.95          | baseline       |
| MOT17-10-SDP   | 654            | 4.16                         | 157.2                    | 56.99          | 76.47          | 71.28          | baseline       |
| MOT17-11-SDP   | 900            | 2.61                         | 344.8                    | 70.58          | 80.09          | 80.24          | baseline       |
| MOT17-13-SDP   | 750            | 4.58                         | 163.8                    | 67.38          | 86.83          | 81.29          | baseline       |
| Combined       | 5,316          | 68.23 (总和)                 | 77.9 (平均)              | 73.65          | 86.08          | 84.79          | baseline       |

### baseline+优化1（加cmc） 

优化1为只用kalman.py脚本，ucmc.py脚本用原来的base版本

| ***\*序列 (Seq)\**** | ***\*帧数 (Frames)\**** | ***\*Time cost (s)\**** | ***\*FPS (Frames/s)\**** | ***\*HOTA\**** | ***\*MOTA\**** | ***\*IDF1\**** | ***\*备注\**** |
| -------------------- | ----------------------- | ----------------------- | ------------------------ | -------------- | -------------- | -------------- | -------------- |
| MOT17-02-SDP         | 600                     | 8.03                    | 74.7                     | 57.31          | 69.49          | 67.24          | Optimized I    |
| MOT17-04-SDP         | 1050                    | 24.79                   | 42.4                     | 83.21          | 94.42          | 92.69          | Optimized I    |
| MOT17-05-SDP         | 837                     | 1.21                    | 691.7                    | 64.98          | 80.69          | 79.16          | Optimized I    |
| MOT17-09-SDP         | 525                     | 0.89                    | 590.0                    | 63.89          | 88.92          | 73.39          | Optimized I    |
| MOT17-10-SDP         | 654                     | 2.80                    | 233.6                    | 50.95          | 73.49          | 61.78          | Optimized I    |
| MOT17-11-SDP         | 900                     | 1.88                    | 478.7                    | 66.48          | 78.46          | 74.24          | Optimized I    |
| MOT17-13-SDP         | 750                     | 3.11                    | 241.2                    | 62.00          | 84.18          | 72.47          | Optimized I    |
| Combined             | 5316                    | 42.8 (总和)             | 124.2 (平均)             | 70.34          | 84.39          | 79.76          | Optimized I    |

### baseline+优化1+优化2（加cmc） 

优化1+优化2为用优化后的kalman.py和ucmc.py脚本

| ***\*序列 (Seq)\**** | ***\*帧数 (Frames)\**** | ***\*Time cost (s)\**** | ***\*FPS (Frames/s)\**** | ***\*HOTA\**** | ***\*MOTA\**** | ***\*IDF1\**** | ***\*备注\**** |
| -------------------- | ----------------------- | ----------------------- | ------------------------ | -------------- | -------------- | -------------- | -------------- |
| MOT17-02-SDP         | 600                     | 3.49                    | 172.1                    | 54.90          | 59.87          | 63.69          | Optimized II   |
| MOT17-04-SDP         | 1050                    | 7.02                    | 149.5                    | 83.06          | 93.64          | 92.60          | Optimized II   |
| MOT17-05-SDP         | 837                     | 1.41                    | 593.6                    | 64.05          | 80.83          | 78.19          | Optimized II   |
| MOT17-09-SDP         | 525                     | 0.96                    | 546.9                    | 64.07          | 89.26          | 73.45          | Optimized II   |
| MOT17-10-SDP         | 654                     | 2.31                    | 283.3                    | 52.84          | 74.42          | 64.40          | Optimized II   |
| MOT17-11-SDP         | 900                     | 1.78                    | 505.6                    | 63.37          | 77.77          | 70.40          | Optimized II   |
| MOT17-13-SDP         | 750                     | 2.04                    | 367.6                    | 61.63          | 83.84          | 72.97          | Optimized II   |
| Combined             | 5316                    | 18.0 (总和)             | 302.7 (平均)             | 69.50          | 82.51          | 78.86          | Optimized II   |

### baseline（不加cmc）

| ***\*序列 (Seq)\**** | ***\*帧数 (Frames)\**** | ***\*Time cost (s)\**** | ***\*FPS (Frames/s)\**** | ***\*HOTA\**** | ***\*MOTA\**** | ***\*IDF1\**** | ***\*备注\**** |
| -------------------- | ----------------------- | ----------------------- | ------------------------ | -------------- | -------------- | -------------- | -------------- |
| MOT17-02-SDP         | 600                     | 8.92                    | 67.3                     | 61.53          | 72.48          | 72.65          | baseline       |
| MOT17-04-SDP         | 1050                    | 41.03                   | 25.6                     | 84.52          | 94.90          | 94.63          | baseline       |
| MOT17-05-SDP         | 837                     | 1.42                    | 589.4                    | 62.64          | 78.96          | 74.97          | baseline       |
| MOT17-09-SDP         | 525                     | 1.12                    | 468.8                    | 74.68          | 91.18          | 88.92          | baseline       |
| MOT17-10-SDP         | 654                     | 3.86                    | 169.4                    | 48.74          | 73.60          | 55.78          | baseline       |
| MOT17-11-SDP         | 900                     | 2.32                    | 387.9                    | 67.56          | 79.66          | 75.85          | baseline       |
| MOT17-13-SDP         | 750                     | 4.36                    | 172.0                    | 62.75          | 84.30          | 75.46          | baseline       |
| Combined             | 5316                    | 62.03（总和）           | 268.6（平均）            | 71.98          | 85.22          | 81.82          | baseline       |

 ### baseline+优化1（不加cmc） 

| ***\*序列 (Seq)\**** | ***\*帧数 (Frames)\**** | ***\*Time cost (s)\**** | ***\*FPS (Frames/s)\**** | ***\*HOTA\**** | ***\*MOTA\**** | ***\*IDF1\**** | ***\*备注\**** |
| -------------------- | ----------------------- | ----------------------- | ------------------------ | -------------- | -------------- | -------------- | -------------- |
| MOT17-02-SDP         | 600                     | 5.72                    | 104.9                    | 57.74          | 69.70          | 67.66          | Optimized I    |
| MOT17-04-SDP         | 1050                    | 24.05                   | 43.7                     | 83.21          | 94.42          | 92.69          | Optimized I    |
| MOT17-05-SDP         | 837                     | 1.25                    | 669.6                    | 63.68          | 80.21          | 76.93          | Optimized I    |
| MOT17-09-SDP         | 525                     | 0.74                    | 709.5                    | 63.89          | 88.92          | 73.39          | Optimized I    |
| MOT17-10-SDP         | 654                     | 2.52                    | 259.5                    | 48.12          | 72.93          | 57.53          | Optimized I    |
| MOT17-11-SDP         | 900                     | 1.64                    | 548.8                    | 66.50          | 78.42          | 74.37          | Optimized I    |
| MOT17-13-SDP         | 750                     | 2.66                    | 281.9                    | 62.39          | 84.40          | 73.75          | Optimized I    |
| Combined             | 5316                    | 38.58 (总和)            | 374.0 (平均)             | 70.13          | 84.35          | 79.38          | Optimized I    |

### baseline+优化1+优化2（不加cmc） 

| ***\*序列 (Seq)\**** | ***\*帧数 (Frames)\**** | ***\*Time cost (s)\**** | ***\*FPS (Frames/s)\**** | ***\*HOTA\**** | ***\*MOTA\**** | ***\*IDF1\**** | ***\*备注\**** |
| -------------------- | ----------------------- | ----------------------- | ------------------------ | -------------- | -------------- | -------------- | -------------- |
| MOT17-02-SDP         | 600                     | 2.45                    | 244.9                    | 55.08          | 60.47          | 63.34          | Optimized II   |
| MOT17-04-SDP         | 1050                    | 4.93                    | 213.0                    | 83.06          | 93.64          | 92.60          | Optimized II   |
| MOT17-05-SDP         | 837                     | 0.87                    | 962.1                    | 63.37          | 79.57          | 76.68          | Optimized II   |
| MOT17-09-SDP         | 525                     | 0.62                    | 846.8                    | 64.06          | 89.26          | 73.45          | Optimized II   |
| MOT17-10-SDP         | 654                     | 1.79                    | 365.9                    | 47.65          | 72.86          | 58.01          | Optimized II   |
| MOT17-11-SDP         | 900                     | 1.37                    | 657.7                    | 66.69          | 77.52          | 74.50          | Optimized II   |
| MOT17-13-SDP         | 750                     | 1.62                    | 463.0                    | 60.57          | 82.75          | 72.73          | Optimized II   |
| Combined             | 5316                    | 13.65（总和）           | 536.0（平均）            | 69.19          | 82.22          | 78.33          | Optimized II   |

## demo展示

![](demo/demo.gif)

#### Environment
- Python (3.8 or later)
- PyTorch with CUDA support
- Ultralytics Library
- Download weight file [yolov8x.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) to folder `pretrained`

#### Run the demo

```bash
python demo.py --cam_para demo/cam_para.txt --video demo/demo.mp4
```
The file `demo/cam_para.txt` is the camera parameters estimated from a single image. The code of this tool is released.  For specific steps, please refer to the Get Started.


## Get Started
- Install the required dependency packages 

```bash
pip install -r requirements.txt
```

- Run UCMCTrack on the MOT17 test dataset, and the tracking results are saved in the folder `output/mot17/test`

```bash
. run_mot17_test.sh
```

- Run UCMCTrack on the MOT17 validation dataset without cmc and evaluate performance metrics such as IDF1, HOTA, and MOTA locally

```bash
. run_mot17_val.bat
```

- Run UCMCTrack on the MOT17 validation dataset and evaluate performance metrics such as IDF1, HOTA, and MOTA locally

```bash
. run_mot17_val.bat
```

