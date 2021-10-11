# 重跑jcy学长论文中的实验,验证有效性
# 全规模太慢了(why???)
# 只切开头
cd ..
num=${0#*./}
num=${num%.sh}
mkdir experiment/$num
python -u train_the_model.py -d experiment/$num --workers 0 -e 1000 --lr 1e-6 --weight_decay 0 -g 1 -b 10 --datatype cqt --train_scale so_short --test_scale so_short --test_source shs_wxy --train_cut front > experiment/$num/out.txt

# lr: 1e-5 loss->5/30 (原0.3.1和0.3.1.1)
# lr: 5e-6 loss->10 (0.3.2)
# lr: 1e-6 loss->30 (0.3.1)
# lr: 1e-7 loss->7.5 (0.3.3)