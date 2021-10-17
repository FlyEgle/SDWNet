# Private NTIRE codebase

# 1. Intro
The code is for NTIRE 2021 deblur track2.

# 2. Folder Structure
```
  ---SR_NTIRE2021
  |
  |- config
  |    |- model.yaml                            -> Model all traninig hyparameters with data log.
  |    |-Config.py                              -> Translate the config file to dict.
  |- data
  |    |- vanilar_dataset.py                    -> The dataset for build the LR & HR images.
  |    |- utils.py                              -> Utils for get patch and calculate the model metrics.
  |    |- augments.py                           -> Augment method for LR & HR images.
  |- model
  |    - NTIRE2021_Deblur
  |        - uniA_ELU
  |            |- layerlib_stage1               -> Model module.
  |            |- model_stage1_Upsample_Deep.py -> Main model.
  |- loss
  |   |- gendrator_loss.py                      -> Loss function define.
  |- optim
  |   |- optimizer.py                           -> Optimizer function define.
  |- train.py                                   -> Training.
  |- final_train.sh                             -> Training shell.
  |- inference_ddp.py                           -> Inference.
  |- inference_ddp.sh                           -> Inference shell.

```
# 3. Training
- Crop the src Training LR and HR images to 480x480 by sliding window which step is 240, so we got 24 patchs form one 720x1280 images both LR and HR.
- Training the model with the 416 x 416 size, use randomcrop, RGB shuffle, horizon flip, rotate and so on.
- Normalize the images to Tensor with 255 but not 1. which without process the mean and std.
```shell
python -W ignore train.py \
--config_file $config_folder \
--dist-url 'tcp://127.0.0.1:8888' \
--dist-backend 'nccl' \
--multiprocessing-distributed=1 \
--world-size=1 \
--rank=0 \
```
# 4. Inference
- Inference the src LR images and get the SR images
```shell
python -W ignore inference_ddp.py \
--config_file $config_folder \
--dist-url 'tcp://127.0.0.1:8989' \
--dist-backend 'nccl' \
--multiprocessing-distributed=1 \
--world-size=1 \
--rank=0 \
```

# 5. Calculate
- Calculate the PSNR and SSIM
```
python utils/calc_psnr_ssim_official.py
```