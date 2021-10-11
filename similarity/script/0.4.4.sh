# 重新设计了gdoras_train_triplet_short和gdoras_train_short
# 小规模：只有两首歌
# 训练集和测试集完全一样
# 希望过拟合
# 在几个固定的地方切，期望会较慢收敛(比每400切一次的慢一点)
cd ..
num=${0#*./}
num=${num%.sh}
mkdir experiment/$num
python -u train_the_model.py -d experiment/$num --workers 0 -e 500 --lr 1e-5 --weight_decay 1e-5 -g 2 -b 6 --train_scale short --test_scale short --test_source gdoras_train --train_cut semi-random-200 > experiment/$num/out.txt

# 确实，训练500epoch后感觉到MAP开始上升了，但是过程非常艰难