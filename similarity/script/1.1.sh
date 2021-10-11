# jcy学长的模型，先在整个训练集、测试集上运行一下试试
cd ..
num=${0#*./}
num=${num%.sh}
mkdir experiment/$num
python -u train_the_model.py -d experiment/$num --workers 0 -e 100 --lr 1e-5 --weight_decay 1e-5 -g 2,3 -b 30 --train_scale long --test_scale middle --test_source gdoras_test > experiment/$num/out.txt