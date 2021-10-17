# cd /data/jiangmingchao/data/SR_NTIRE2021;
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/unia_255_no_norm.yaml" \
# --dist-url 'tcp://127.0.0.1:9999' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \

# cd /data/jiangmingchao/data/SR_NTIRE2021;
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/unia_255_norm.yaml" \
# --dist-url 'tcp://127.0.0.1:8888' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \

# cd /data/jiangmingchao/data/SR_NTIRE2021;
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/unia_1_no_norm.yaml" \
# --dist-url 'tcp://127.0.0.1:9999' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \

# cd /data/jiangmingchao/data/SR_NTIRE2021;
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/unia_1_norm.yaml" \
# --dist-url 'tcp://127.0.0.1:8888' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \


# resoulution
# 48x48
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/resolution/unia_255_no_norm_48x48.yaml" \
# --dist-url 'tcp://127.0.0.1:8888' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \

# 96 x 96
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/resolution/unia_255_no_norm_96x96.yaml" \
# --dist-url 'tcp://127.0.0.1:9999' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \


# 192 x 192
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/resolution/unia_255_no_norm_192x192.yaml" \
# --dist-url 'tcp://127.0.0.1:8888' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \

# 256x256
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/resolution/unia_255_no_norm_256x256.yaml" \
# --dist-url 'tcp://127.0.0.1:9999' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \

# 512x512
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/resolution/unia_255_no_norm_512x512.yaml" \
# --dist-url 'tcp://127.0.0.1:9999' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \

# 96X96 SGD cosine
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/hyperparameters/unia_255_no_norm_96x96_SGD_cosine.yaml" \
# --dist-url 'tcp://127.0.0.1:8888' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \


# 96x96 adam cosine
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/hyperparameters/unia_255_no_norm_96x96_adam_cosine.yaml" \
# --dist-url 'tcp://127.0.0.1:9999' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \

# 96X96 adamw cosine
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/hyperparameters/unia_255_no_norm_96x96_adamw_cosine.yaml" \
# --dist-url 'tcp://127.0.0.1:8888' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \

# 96x96 adam cosine cutblur
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/data_aug/unia_255_no_norm_96x96_adam_cosine_cutblur.yaml" \
# --dist-url 'tcp://127.0.0.1:8888' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \

# 96x96 adam cosine cutblur rgb
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/data_aug/unia_255_no_norm_96x96_adam_cosine_cutblur_rgb.yaml" \
# --dist-url 'tcp://127.0.0.1:9999' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \

# 480x480 adamw cosine cropx24 90 epoch
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/crop/unia_255_no_norm_480x480_adamw_cosine_cropx24_0.1.yaml" \
# --dist-url 'tcp://127.0.0.1:9999' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \

# 480x480 adamw cosine cropx24 0.2 90epoch
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/crop/unia_255_no_norm_480x480_adamw_cosine_cropx24_0.2.yaml" \
# --dist-url 'tcp://127.0.0.1:9999' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \

# TODO
# 480x480 adamw cosine cropx24 psnr 60 epoch
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/crop/unia_255_no_norm_480x480_adamw_cosine_cropx24_psnr.yaml" \
# --dist-url 'tcp://127.0.0.1:9999' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \


# 480x480 adamw cosine CL1 with ssim
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/loss/unia_255_no_norm_480x480_adamw_cosine_l1loss_ssim.yaml" \
# --dist-url 'tcp://127.0.0.1:9999' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \

# 480x480 adamw cosine CL1 with ms-ssim
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/loss/unia_255_no_norm_480x480_adamw_cosine_cl1loss_ms_ssim.yaml" \
# --dist-url 'tcp://127.0.0.1:8888' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \

# 480x480 adamw cosine CL1 with ms-ssim no acc
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/loss/unia_255_no_norm_480x480_adamw_cosine_cl1loss_ssim_no_acc.yaml" \
# --dist-url 'tcp://127.0.0.1:8888' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \

# 480x480 adamw cosine CL1 with ms-ssim no acc
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/loss/unia_255_no_norm_96x96_addamw_cosine_cl1loss_ssim_acc_4.yaml" \
# --dist-url 'tcp://127.0.0.1:9999' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \



# 480x480 adamw cosine CL1 with ms-ssim no acc do not devide step 4
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/loss/unia_255_no_norm_96x96_adamw_cosine_l1closs_ssim_acc_no_step_4.yaml" \
# --dist-url 'tcp://127.0.0.1:9999' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \


# 480x480 adamw cosine CL1 with ms-ssim no acc do not devide step 2
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/loss/unia_255_no_norm_480x480_adamw_cosine_l1closs_simm_acc_no_step_2.yaml" \
# --dist-url 'tcp://127.0.0.1:8888' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \


# 480x480 adamw cosine CL1 with ssim elu
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/model/unia_no_norm_255_elu_cl1loss_ssim_480x480.yaml" \
# --dist-url 'tcp://127.0.0.1:8888' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \

# 480x480 adamw cosine CL1 with ssim elu 150 epoch
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/model/unia_no_norm_255_elu_cl1loss_ssim_480x480_150.yaml" \
# --dist-url 'tcp://127.0.0.1:8888' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \

# 480x480 adamw cosine CL1 with ssim elu 90 epoch
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/model/unia_no_norm_255_elu_cl1loss_ssim_480x480_90.yaml" \
# --dist-url 'tcp://127.0.0.1:8888' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \


# 480x480 adamw cosine CL1 with ssim elu 90 epoch
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/model/unia_wide_no_norm_255_elu_cl1loss_ssim_480x480_120.yaml" \
# --dist-url 'tcp://127.0.0.1:8888' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \


# 96x96 adamw cosine CL1 with ssim elu lrx2 lars
# cd /data/jiangmingchao/data/SR_NTIRE2021;
# python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/model/wide_96x96_lRX2_LARS.yaml" \
# --dist-url 'tcp://127.0.0.1:8888' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \

# 416x416 adamw cosine CL1 with ssim elu lrx2 lars

cd /data/jiangmingchao/data/SR_NTIRE2021;
python -W ignore train.py \
--config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/model/wide_416x416_lRX2_LARS.yaml" \
--dist-url 'tcp://127.0.0.1:8888' \
--dist-backend 'nccl' \
--multiprocessing-distributed=1 \
--world-size=1 \
--rank=0 \