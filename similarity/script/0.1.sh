# 重新设计了gdoras_train_triplet_short和gdoras_train_short
# 小规模：只有两首歌
# 训练集和测试集完全一样
# 希望过拟合
cd ..
num=${0#*./}
num=${num%.sh}
mkdir experiment/$num
python -u train_the_model.py -d experiment/$num --workers 0 -e 1000 --lr 1e-6 --weight_decay 1e-5 -g 0 -b 6 --train_scale so_short --test_scale so_short --test_source gdoras_train --train_cut random > experiment/$num/out.txt