# 重跑jcy学长论文中的实验,验证有效性
# 全规模太慢了(why???)
cd ..
num=${0#*./}
num=${num%.sh}
mkdir experiment/$num
python -u train_the_model.py -d experiment/$num --workers 0 -e 1000 --lr 1e-5 --weight_decay 1e-5 -g 0 -b 5 --datatype cqt --train_scale so_short --test_scale so_short --test_source shs_wxy --train_cut random > experiment/$num/out.txt