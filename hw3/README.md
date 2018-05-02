# DLCV 2018 Spring HW3
## Semantic Segmentation
### Inference
- Baseline model (VGG16_FCN32s) [1]
```
bash hw3.sh <testing images directory> <output images directory>
```
- Best model (VGG16_FCN8s) [1]
```
bash hw3_best.sh <testing images directory> <output images directory>
```
- Dilated model (VGG16_dilated) [2]
```
bash hw3_dilated.sh <testing images directory> <output images directory>
```
[Note] The above scripts will download trained model (~ 1GB each) from the internet.

### Training
- Baseline model (VGG16_FCN32s) [1]
```
python3 train.py --train_path <training images directory> --valid_path <validation images directory> --vgg_path <vgg16 pretrain weight directory> --dir_name <folder name for the saved models> --result_path <result directory>
```
Other parameters:
```
--fcn_stride: 32 for FCN32s; 8 for FCN8s, default = 32
--num_epoch: Number of epochs, default = 50
--batch_size: Batch size, default = 32
--learning_rate: Initial learning rate, default = 1e-5
--patience: Patience for early stopping, default = 10
```

- Best model (VGG16_FCN32s) [1]
```
python3 train.py --train_path <training images directory> --valid_path <validation images directory> --vgg_path <vgg16 pretrain weight directory> --dir_name <folder name for the saved models> --result_path <result directory> --fcn_stride 8 --batch_size 16
```

- Dilated model (VGG16_dilated) [2]
```
python3 train_dilated.py --train_path <training images directory> --valid_path <validation images directory> --vgg_path <vgg16 pretrain weight directory> --dir_name <folder name for the saved models> --result_path <result directory> --batch_size 16
```

### Evaluation (mean IoU), provided by TAs
```
python3 mean_iou_evaluate.py -g <ground_truth_directory> -p <prediction_directory>
```

### Mean IoU of the best model in each case
| VGG16_FCN32s | VGG16_FCN8s | VGG16_dilated |
|--------------|-------------|---------------|
| 0.67754      | 0.70554     | 0.69120       |

### References
[1] Jonathan Long, Evan Shelhamer, and Trevor Darrell, "Fully convolutional networks for semantic segmentation". In CVPR, 2015.  
[2] Fisher Yu and Vladlen Koltun, "Multi-scale context aggregation by dilated convolutions". In ICLR, 2016.  
[3] Tensorflow implementation: https://github.com/sagieppel/Fully-convolutional-neural-network-FCN-for-semantic-segmentation-Tensorflow-implementation  
[4] Pre-trained vgg16: https://drive.google.com/file/d/0B6njwynsu2hXZWcwX0FKTGJKRWs/view?usp=sharing or ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy
