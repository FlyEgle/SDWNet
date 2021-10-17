# cd /data/jiangmingchao/data/SR_NTIRE2021;
# python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/crop/ALLDATA.yaml" \
# --dist-url 'tcp://127.0.0.1:8888' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \

# cd /data/jiangmingchao/data/SR_NTIRE2021;
# python -W ignore train.py \
# --config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/wavelet_two_stage/finetune_all_data.yaml" \
# --dist-url 'tcp://127.0.0.1:8888' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed=1 \
# --world-size=1 \
# --rank=0 \


cd /data/jiangmingchao/data/SR_NTIRE2021;
python -W ignore train.py \
--config_file "/data/jiangmingchao/data/SR_NTIRE2021/config/wavelet_two_stage/finetune_all_data.yaml" \
--dist-url 'tcp://127.0.0.1:8888' \
--dist-backend 'nccl' \
--multiprocessing-distributed=1 \
--world-size=1 \
--rank=0 \