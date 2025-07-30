import torch
import torch.nn as nn
import ResNet
import lmmd  # 确保MMD_loss在lmmd.py中

class DSAN(nn.Module):
    def __init__(self, num_classes=7, bottle_neck=True):
        super(DSAN, self).__init__()
        self.feature_layers = ResNet.resnet50(True)
        self.mmd_loss = lmmd.MMD_loss()  # 使用MMD损失
        self.bottle_neck = bottle_neck
        
        if bottle_neck:
            self.bottle = nn.Linear(2048, 256)
            self.cls_fc = nn.Linear(256, num_classes)
        else:
            self.cls_fc = nn.Linear(2048, num_classes)

    def forward(self, source, target):
        # 提取源域特征
        source_feat = self.feature_layers(source)
        if self.bottle_neck:
            source_feat = self.bottle(source_feat)
        s_pred = self.cls_fc(source_feat)
        
        # 提取目标域特征
        target_feat = self.feature_layers(target)
        if self.bottle_neck:
            target_feat = self.bottle(target_feat)
        
        # 计算MMD损失
        loss_mmd = self.mmd_loss.get_loss(source_feat, target_feat)
        
        # 只返回2个值：分类预测和MMD损失
        return s_pred, loss_mmd

    def predict(self, x):
        x = self.feature_layers(x)
        if self.bottle_neck:
            x = self.bottle(x)
        return self.cls_fc(x)