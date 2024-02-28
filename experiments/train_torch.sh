export PYTHONPATH=../:$PYTHONPATH
CUDA_VISIBLE_DEVICES=$2 python -m torch.distributed.launch --master_port $3 --nproc_per_node=$1 ../tools/train_val.py
