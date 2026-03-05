import torchvision
from torch import nn
from torch.nn import init
from models.utils import pooling
        
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

class ResNet50(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet50.layer4[0].conv2.stride = (1, 1)
            resnet50.layer4[0].downsample[0].stride = (1, 1) 

        # backbone till layer4
        self.base = nn.Sequential(*list(resnet50.children())[:-2])

        # pooling
        if config.MODEL.POOLING.NAME == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool2d(1)
        elif config.MODEL.POOLING.NAME == 'max':
            self.globalpooling = nn.AdaptiveMaxPool2d(1)
        elif config.MODEL.POOLING.NAME == 'gem':
            self.globalpooling = pooling.GeMPooling(p=config.MODEL.POOLING.P)
        elif config.MODEL.POOLING.NAME == 'maxavg':
            self.globalpooling = pooling.MaxAvgPooling()
        else:
            raise KeyError("Invalid pooling: '{}'".format(config.MODEL.POOLING.NAME))

        self.bn = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

        # ⚠️ 新增：占位，方便 test.py 检查 hasattr(model, "fmap")
        self.fmap = None
        
    def forward(self, x):
        # 最后一层特征图 (B,C,H,W)
        feat_map = self.base(x)
        self.fmap = feat_map             # 保存下来供 test 可视化使用

        # 池化 + BN
        x = self.globalpooling(feat_map)
        x = x.view(x.size(0), -1)
        f = self.bn(x)

        return f
