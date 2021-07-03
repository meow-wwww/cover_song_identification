import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', help='目标存储目录')
parser.add_argument('--lr', type=float, help='学习率')
parser.add_argument('--saved_model', help='以训练过的模型')
parser.add_argument('-e', '--epoch', type=int, help='有几个epoch')

args = parser.parse_args()

print(args.save_dir, args.lr, args.saved_model, args.epoch)
print(type(args.save_dir), type(args.lr), type(args.saved_model), type(args.epoch))