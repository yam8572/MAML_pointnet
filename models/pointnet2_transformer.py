import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation
import torch
class MyEncoderLayer(nn.Module):
    def __init__(self, d_model:int, n_head:int) -> None:
        super(MyEncoderLayer, self).__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head,batch_first=True)

    def forward(self, input:torch.Tensor):
        xx = self.encoder(input)
        return xx
    pass

class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 9, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.my_encoder1 = MyEncoderLayer(256, 8)
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.my_encoder2 = MyEncoderLayer(16, 8)
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)


        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:] # PointNet++ 只有使用 XYZ 去訓練

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print(l1_xyz.size(), l1_points.size()) torch.Size([16, 3, 1024]) torch.Size([16, 96, 1024])
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print(l2_xyz.size(), l2_points.size()) torch.Size([16, 3, 256]) torch.Size([16, 256, 256])
        l2_points = self.my_encoder1(l2_points)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # print(l3_xyz.size(), l3_points.size()) torch.Size([16, 3, 64]) torch.Size([16, 512, 64])
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        # print(l4_xyz.size(), l4_points.size()) torch.Size([16, 3, 16]) torch.Size([16, 1024, 16])
        l4_points = self.my_encoder2(l4_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        # print(l3_points.size()) torch.Size([16, 256, 64])
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # print(l2_points.size()) torch.Size([16, 256, 256])
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # print(l1_points.size()) torch.Size([16, 128, 1024])
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        # print(l0_points.size()) torch.Size([16, 128, 1024])

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        # print(x.size()) torch.Size([16, 128, 1024])
        x = self.conv2(x)
        # print(x.size()) torch.Size([16, 13, 1024])
        x = F.log_softmax(x, dim=1)
        # print(x.size()) torch.Size([16, 13, 1024])
        x = x.permute(0, 2, 1)
        # print(x.size()) torch.Size([16, 1024, 13])
        return x, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))