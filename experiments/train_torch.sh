export PYTHONPATH=../:$PYTHONPATH
CUDA_VISIBLE_DEVICES=$3 python -m torch.distributed.launch --master_port $4 --nproc_per_node=$2 ../tools/train_val.py --config $1
