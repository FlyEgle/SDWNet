cd /data/jiangmingchao/data/SR_NTIRE2021;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore inference_ddp.py \
--config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/inference/paper/goprol_base.yaml" \
--dist-url 'tcp://127.0.0.1:9898' \
--dist-backend 'nccl' \
--multiprocessing-distributed=1 \
--world-size=1 \
--rank=0 \