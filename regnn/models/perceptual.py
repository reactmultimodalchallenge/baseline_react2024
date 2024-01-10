import torch
import torch.nn as nn
import torchvision
from .swin_transformer import SwinTransformer
from .torchvggish.vggish import VGGish
from .MULTModel import MULTModel
# net_to_call(embed_dim = 96, depths = [2, 2, 6, 2], num_heads = [3, 6, 12, 24], window_size = 7, drop_path_rate = 0.2)

class VideoProcessor(nn.Module):
    def __init__(self, base_type='Swin', frame_size=50, pretrained=False):
        super().__init__()
        self.frame_size = frame_size
        if base_type == 'Swin':
            self.model = SwinTransformer(embed_dim = 96, depths = [2, 2, 6, 2], num_heads = [3, 6, 12, 24],
                                         window_size = 7, drop_path_rate = 0.2)
        else:
            raise KeyError ('Invalid model types')
        if pretrained:
            self.model.load_state_dict(torch.load(r"/scratch/recface/hz204/react_data/pretrained/swin_fer.pth",
                                                  map_location='cpu'))
            # ["model"])

    def forward(self, inputs):
        N, C, H, W = inputs.shape
        # print(N, C, H, W)
        assert N==self.frame_size
        # outputs = self.model(inputs)
        outputs = self.model.forward_features(inputs)
        assert outputs.shape[0]==N
        return outputs

class AudioProcessor(nn.Module):
    def __init__(self, base_type='VGGish', ):
        super().__init__()
        if base_type == 'VGGish':
            # 如果输入是tensor，preprocess=False,如果输入是array/str,preprocess=True
            self.model = VGGish(preprocess=False)
        else:
            raise KeyError('Invalid model types')

    def forward(self, file_name):
        outputs = self.model(file_name)
        return outputs


class PercepProcessor(nn.Module):
    def __init__(self, v_type='Swin', a_type='VGGish', only_fuse=False):
        super().__init__()
        if not only_fuse:
            self.v_processor = VideoProcessor(base_type=v_type)
            self.a_processor = AudioProcessor(base_type=a_type)
        self.fuse_model = MULTModel()
        self.only_fuse = only_fuse

    # B T D
    # video_inputs: B, T, 768
    # audio_inputs: B, T, 128
    # outputs: B, T, 64
    def forward(self, video_inputs, audio_inputs):
        if not self.only_fuse:
            video_inputs = self.v_processor(video_inputs)
            audio_inputs = self.a_processor(audio_inputs)
            video_inputs = video_inputs.unsqueeze(0)
            audio_inputs = audio_inputs.unsqueeze(0)

        # print('*'*50)
        # print('V-SEQ:', v_seq.shape)
        # print('A-SEQ:', a_seq.shape)
        # print('*' * 50)
        # print('video：', video_inputs.shape)
        # print('audio：', audio_inputs.shape)
        # fused_feature: T, B, 64
        fused_feature = self.fuse_model(video_inputs, audio_inputs)

        return fused_feature.transpose(1, 0)


