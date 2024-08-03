export PYTHONPATH=../:$PYTHONPATH
CUDA_VISIBLE_DEVICES=$3 python -m torch.distributed.launch --nproc_per_node=$2 ../tools/train_val.py -e --config $1
