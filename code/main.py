import torch
import torch.nn.functional as F
import math
import argparse
import numpy as np
import os

from DSAN import DSAN
import data_loader


def load_data(root_path, src, tar, batch_size):
    # 检查源域和目标域路径是否存在
    src_path = os.path.join(root_path, src)
    tar_path = os.path.join(root_path, tar)
    
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"源域路径不存在: {src_path}")
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"目标域路径不存在: {tar_path}")
    
    # 获取有效的类别文件夹（过滤隐藏文件夹）
    def get_valid_classes(path):
        return [d for d in os.listdir(path) 
                if os.path.isdir(os.path.join(path, d)) and not d.startswith('.')]
    
    src_classes = get_valid_classes(src_path)
    tar_classes = get_valid_classes(tar_path)
    
    if not src_classes:
        raise ValueError(f"源域路径 {src_path} 下没有找到有效的类别文件夹")
    if not tar_classes:
        raise ValueError(f"目标域路径 {tar_path} 下没有找到有效的类别文件夹")
    
    # 打印找到的类别
    print(f"找到 {len(src_classes)} 个源域类别: {src_classes}")
    print(f"找到 {len(tar_classes)} 个目标域类别: {tar_classes}")
    
    # 检查类别数是否匹配
    if len(src_classes) != len(tar_classes):
        print(f"警告: 源域类别数 ({len(src_classes)}) 与目标域类别数 ({len(tar_classes)}) 不匹配")
        print(f"源域类别: {src_classes}")
        print(f"目标域类别: {tar_classes}")
    
    kwargs = {'num_workers': 1, 'pin_memory': True}
    loader_src = data_loader.load_training(root_path, src, batch_size, kwargs)
    loader_tar = data_loader.load_training(root_path, tar, batch_size, kwargs)
    loader_tar_test = data_loader.load_testing(root_path, tar, batch_size, kwargs)
    return loader_src, loader_tar, loader_tar_test


def train_epoch(epoch, model, dataloaders, optimizer, args):
    model.train()
    source_loader, target_train_loader, _ = dataloaders
    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len(source_loader)
    
    for i in range(1, num_iter):
        try:
            data_source, label_source = next(iter_source)
        except StopIteration:
            iter_source = iter(source_loader)
            data_source, label_source = next(iter_source)
            
        try:
            data_target, _ = next(iter_target)
        except StopIteration:
            iter_target = iter(target_train_loader)
            data_target, _ = next(iter_target)
            
        data_source, label_source = data_source.cuda(), label_source.cuda()
        data_target = data_target.cuda()

        optimizer.zero_grad()
        # 确保只接收2个返回值
        s_pred, loss_mmd = model(data_source, data_target)
        
        # 计算分类损失
        loss_cls = F.nll_loss(F.log_softmax(s_pred, dim=1), label_source)
        
        # 自适应权重
        lambd = 2 / (1 + math.exp(-10 * (epoch) / args.nepoch)) - 1
        loss = loss_cls + args.weight * lambd * loss_mmd

        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print(f'Epoch: [{epoch:2d}], Loss: {loss.item():.4f}, cls_Loss: {loss_cls.item():.4f}, loss_mmd: {loss_mmd.item():.4f}')


def test(model, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.cuda(), target.cuda()
            pred = model.predict(data)
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(dataloader)
        print(f'Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(dataloader.dataset)} ({100. * correct / len(dataloader.dataset):.2f}%)')
    
    return correct


def get_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='Root path for dataset',
                        default='/root/autodl-tmp/database/FER2013/')
    parser.add_argument('--src', type=str,
                        help='Source domain', default='test')
    parser.add_argument('--tar', type=str,
                        help='Target domain', default='train')
    parser.add_argument('--nclass', type=int,
                        help='Number of classes', default=7)
    parser.add_argument('--batch_size', type=int,
                        help='batch size', default=32)
    parser.add_argument('--nepoch', type=int,
                        help='Total epoch num', default=200)
    parser.add_argument('--lr', type=list, help='Learning rate', default=[0.001, 0.01, 0.01])
    parser.add_argument('--early_stop', type=int,
                        help='Early stoping number', default=30)
    parser.add_argument('--seed', type=int,
                        help='Seed', default=2021)
    parser.add_argument('--weight', type=float,
                        help='Weight for adaptation loss', default=0.5)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--decay', type=float,
                        help='L2 weight decay', default=5e-4)
    parser.add_argument('--bottleneck', type=str2bool,
                        nargs='?', const=True, default=True)
    parser.add_argument('--log_interval', type=int,
                        help='Log interval', default=10)
    parser.add_argument('--gpu', type=str,
                        help='GPU ID', default='0')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    
    # 打印配置信息
    print("===== 配置信息 =====")
    print(f"源域路径: {os.path.join(args.root_path, args.src)}")
    print(f"目标域路径: {os.path.join(args.root_path, args.tar)}")
    print(f"类别数: {args.nclass}")
    print("===================")
    
    # 设置随机种子
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 加载数据
    try:
        dataloaders = load_data(args.root_path, args.src, args.tar, args.batch_size)
        print("数据加载成功！")
    except Exception as e:
        print(f"数据加载失败: {e}")
        exit(1)

    # 初始化模型
    model = DSAN(num_classes=args.nclass, bottle_neck=args.bottleneck).cuda()
    
    # 优化器设置
    if args.bottleneck:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.bottle.parameters(), 'lr': args.lr[1]},
            {'params': model.cls_fc.parameters(), 'lr': args.lr[2]},
        ], lr=args.lr[0], momentum=args.momentum, weight_decay=args.decay)
    else:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': args.lr[1]},
        ], lr=args.lr[0], momentum=args.momentum, weight_decay=args.decay)

    # 训练和测试
    correct = 0
    stop = 0

    for epoch in range(1, args.nepoch + 1):
        stop += 1
        
        # 学习率衰减
        for index, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = args.lr[index] / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75)

        print(f"\n=== Epoch {epoch}/{args.nepoch} ===")
        train_epoch(epoch, model, dataloaders, optimizer, args)
        t_correct = test(model, dataloaders[-1])
        
        # 保存最佳模型
        if t_correct > correct:
            correct = t_correct
            stop = 0
            torch.save(model, 'model.pkl')
            print(f"模型已保存，准确率: {100. * correct / len(dataloaders[-1].dataset):.2f}%")
            
        print(f'{args.src}-{args.tar}: max correct: {correct} max accuracy: {100. * correct / len(dataloaders[-1].dataset):.2f}%\n')

        # 早停机制
        if stop >= args.early_stop:
            print(f'达到早停条件，最终测试准确率: {100. * correct / len(dataloaders[-1].dataset):.2f}%')
            break