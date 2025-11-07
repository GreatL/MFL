# weibo
python train.py --dataset weibo --dim 64 --lr 0.0006 --num-layers 3 --cuda 0
# twitter
python train.py --dataset twitter --dim 16 --lr 0.0003 --num-layers 2 --cuda 0
# Fakeddit
python train.py --dataset Fakeddit --dim 64 --lr 0.0003 --num-layers 2 --cuda 0
# FineFake
python train.py --dataset FineFake --dim 32 --lr 0.0003 --num-layers 2 --cuda 0
