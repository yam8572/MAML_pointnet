import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer


class get_model(nn.Module):
    def __init__(self, num_class):
        super(get_model, self).__init__()
        self.k = num_class
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=9)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        # print(x.size()) torch.Size([16, 9, 4096])
        x, trans, trans_feat = self.feat(x)
        # print(x.size()) torch.Size([16, 1088, 4096])
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.size()) torch.Size([16, 512, 4096])
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.size()) torch.Size([16, 256, 4096])
        x = F.relu(self.bn3(self.conv3(x)))
        # print(x.size()) torch.Size([16, 128, 4096])
        x = self.conv4(x) 
        # print(x.size()) torch.Size([16, 13, 4096])
        x = x.transpose(2,1).contiguous()
        # print(x.size()) torch.Size([16, 4096, 13])
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        # print(x.size()) torch.Size([65536, 13])
        x = x.view(batchsize, n_pts, self.k)
        # print(x.size(), trans_feat.size()) torch.Size([16, 4096, 13]) torch.Size([16, 64, 64])
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, weight):
        loss = F.nll_loss(pred, target, weight = weight)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


if __name__ == '__main__':
    model = get_model(13)
    xyz = torch.rand(12, 3, 2048)
    (model(xyz))