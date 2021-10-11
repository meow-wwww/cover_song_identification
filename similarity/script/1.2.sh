# long数据集太大，都换成middle的，都用训练集（过拟合）
cd ..
num=${0#*./}
num=${num%.sh}
mkdir experiment/$num
python -u train_the_model.py -d experiment/$num --workers 0 -e 100 --lr 1e-3 --weight_decay 1e-5 -g 3 -b 20 --train_scale middle --test_scale middle --test_source gdoras_test > experiment/$num/out.txt
