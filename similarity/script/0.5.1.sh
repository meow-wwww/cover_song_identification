# 训练集只有两首歌；测试集大概10首
# 训练集和测试集不一样
# 在几个固定的地方切，期望会较快收敛
cd ..
num=${0#*./}
num=${num%.sh}
mkdir experiment/$num
python -u train_the_model.py -d experiment/$num --workers 0 -e 1000 --lr 1e-5 --weight_decay 1e-5 -g 0,3 -b 20 --train_scale short --test_scale short --test_source gdoras_test --train_cut semi-random-600 > experiment/$num/out.txt

# 