cd /data/jiangmingchao/data/SR_NTIRE2021;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore train_goprol.py \
--config_file="/data/jiangmingchao/data/SR_NTIRE2021/config/paper_prepare/deblur_param/goprol_pretrain_500_from_new_new_best.yaml" \
--dist-url 'tcp://127.0.0.1:7878' \
--dist-backend 'nccl' \
--multiprocessing-distributed=1 \
--world-size=1 \
--rank=0 \