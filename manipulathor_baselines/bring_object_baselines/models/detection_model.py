
import torch
# from .feature_learning import FeatureLearnerModule
# from utils.net_util import _upsample_add, upshuffle, upshufflenorelu, combine_block_w_do
from manipulathor_baselines.bring_object_baselines.models.feature_extractor_models import FeatureLearnerModule
from manipulathor_utils.debugger_util import ForkedPdb
from manipulathor_utils.net_utils import combine_block_w_do, upshuffle, upshufflenorelu, _upsample_add
import torch.nn as nn

class ConditionalDetectionModel(nn.Module):
    def __init__(self, ):
        super().__init__()
        # assert args.dropout == 0
        dropout = 0.5

        # self.fixed_feature_weights = args.fixed_feature_weights

        self.pointwise_conv = combine_block_w_do(512 + 512, 64, dropout)

        self.depth_up1 = upshuffle(64, 256, 2, kernel_size=3, stride=1, padding=1)
        self.depth_up2 = upshuffle(256, 128, 2, kernel_size=3, stride=1, padding=1)
        self.depth_up3 = upshuffle(128, 64, 2, kernel_size=3, stride=1, padding=1)
        self.depth_up4 = upshuffle(64, 64, 2, kernel_size=3, stride=1, padding=1)
        self.depth_up5 = upshufflenorelu(64, 2, 2)

        self.feature_extractor = FeatureLearnerModule()
        self.target_category_feature_extractor = FeatureLearnerModule()

        self.eval()



    def forward(self, input):
        self.eval()
        with torch.no_grad():
            assert 'rgb' in input and 'target_cropped_object' in input
            images = input['rgb']
            _ = self.target_category_feature_extractor(input['target_cropped_object'])
            target_category_features = self.target_category_feature_extractor.intermediate_features[-1]



            features = self.feature_extractor(images)
            intermediate_features = self.feature_extractor.intermediate_features
            image_features = intermediate_features[-1]

            full_feature = torch.cat([target_category_features, image_features], dim=1)

            embedded_image_category = self.pointwise_conv(full_feature) #TODO this is 64x7x7 is this a good architecture choice?


            c1, c2, c3, c4, _ = intermediate_features
            d5 = self.depth_up1(embedded_image_category)
            d5_ = _upsample_add(d5, c4)
            d4 = self.depth_up2(d5_)
            d4_ = _upsample_add(d4, c3)
            d3 = self.depth_up3(d4_)
            d3_ = _upsample_add(d3, c2)
            d2 = self.depth_up4(d3_)
            d2_ = _upsample_add(d2, c1)
            object_mask = self.depth_up5(d2_)


            assert object_mask.shape[1] == 2

            output = {
                'object_mask': object_mask
            }

        return output


    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.base_lr)
