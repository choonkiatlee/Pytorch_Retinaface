import torch
import torch.nn as nn
import torchvision
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH

from math import ceil
from itertools import product as product

from utils.box_utils import prior_box, decode_bbox

from typing import *

class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, 
        cfg = None, 
        phase = 'train', 
        calculate_prior_boxes = False, 
        backbone = None,
        ):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()
        self.phase = phase
        # backbone = None
        # if cfg['name'] == 'mobilenet0.25':
        #     backbone = MobileNetV1()
        #     if cfg['pretrain']:
        #         checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
        #         from collections import OrderedDict
        #         new_state_dict = OrderedDict()
        #         for k, v in checkpoint['state_dict'].items():
        #             name = k[7:]  # remove module.
        #             new_state_dict[name] = v
        #         # load params
        #         backbone.load_state_dict(new_state_dict)
        # elif cfg['name'] == 'Resnet50':
        #     import torchvision.models as models
        #     backbone = models.resnet50(pretrained=cfg['pretrain'])

        self.cfg = cfg
        self.calculate_prior_boxes = calculate_prior_boxes

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2) -> nn.ModuleList:
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2) -> nn.ModuleList:
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2) -> nn.ModuleList:
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self,inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        # if self.calculate_prior_boxes:
        #     prior_boxes = self._prior_box((inputs.shape[1], inputs.shape[2]), self.cfg)
        # else:
        #     prior_boxes = None

        # if self.phase == 'train':
        #     output = (bbox_regressions, classifications, ldm_regressions, prior_boxes)
        # else:
        #     output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions, prior_boxes)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)

        return output


class RetinaFaceModified(nn.Module):
    def __init__(self, 
        # cfg = None, 
        phase = 'train', 
        decode_boxes: bool = False, 

        backbone = None,

        # Network Configuration
        return_layers:Dict[str, int] = {'stage1': '1', 'stage2': '2', 'stage3': '3'},
        in_channel:int = 32,
        out_channel:int = 64,
        min_sizes_list: List[Tuple[int,int]] = [(16, 32), (64, 128), (256, 512)],
        steps:List[int] = [8, 16, 32],
        clip: bool = False,
        image_size: Optional[Tuple[int, int]] = None,

        # Decoding Configuration
        variances: List[float] = [0.1, 0.2],
        resize: int = 1,
        confidence_threshold: float = 0.02,
        nms_threshold: float = 0.4,
        ):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFaceModified,self).__init__()
        self.phase = phase

        # self.cfg = cfg
        # Set config
        self.decode_boxes = decode_boxes
        self.min_sizes_list = min_sizes_list
        self.steps = steps
        self.clip = clip
        self.image_size = image_size

        self.variances = variances
        self.resize = resize
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        if self.image_size is not None:
            self.prior_boxes = self._prior_box(torch.tensor(self.image_size).float(), self.min_sizes_list, self.steps, self.clip)
        else:
            self.prior_boxes = None

        self.body = _utils.IntermediateLayerGetter(backbone, return_layers)
        in_channels_stage2 = in_channel
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = out_channel
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=out_channel)
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=out_channel)
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=out_channel)

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    # Traceable version of prior_box
    # @torch.jit.script
    

    def forward(self,inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = (feature1, feature2, feature3)

        bbox_regressions = torch.cat([selected_bbox_head(feature) for selected_bbox_head, feature in zip(self.BboxHead, features)], dim=1)
        classifications = torch.cat([selected_class_head(feature) for selected_class_head, feature in zip(self.ClassHead, features)],dim=1)
        ldm_regressions = torch.cat([selected_landmark_head(feature) for selected_landmark_head, feature in zip(self.LandmarkHead, features)], dim=1)
        
        if (self.image_size is None):
            image_size = (inputs.shape[2], inputs.shape[3])
        else:
            image_size = self.image_size

        boxes = bbox_regressions
        scores = F.softmax(classifications, dim=-1)
        landms = ldm_regressions

        if self.decode_boxes:
            if self.prior_boxes is None:
                prior_boxes = prior_box(torch.tensor(image_size, dtype=torch.float), self.min_sizes_list, self.steps, self.clip)
            else:
                prior_boxes = self.prior_boxes

            output = decode_bbox(
                boxes, 
                scores, 
                landms, 
                prior_boxes, 
                image_size,
                self.variances, 
                self.resize,
                self.confidence_threshold, 
                self.nms_threshold
                )

        else:
            
            if self.phase == 'train':
                output = (bbox_regressions, classifications, ldm_regressions)
            else:
                output = (boxes, scores, landms)

        return output