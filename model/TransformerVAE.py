import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .BasicBlock import ConvBlock, PositionalEncoding, lengths_to_mask, init_biased_mask

class VideoEncoder(nn.Module):
    def __init__(self, img_size=224, feature_dim = 128, device = 'cpu'):
        super(VideoEncoder, self).__init__()

        self.img_size = img_size
        self.feature_dim = feature_dim

        self.Conv3D = ConvBlock(3, feature_dim)
        self.fc = nn.Linear(feature_dim, feature_dim)
        self.device = device


    def forward(self, video):
        """
        input:
        speaker_video_frames x: (batch_size, seq_len, 3, img_size, img_size)

        output:
        speaker_temporal_tokens y: (batch_size, seq_len, token_dim)

        """

        video_input = video.transpose(1, 2)  # B C T H W
        token_output = self.Conv3D(video_input).transpose(1,2)
        token_output = self.fc(token_output) # B T C
        return  token_output



class VAEModel(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int = 256,
                 **kwargs) -> None:
        super(VAEModel, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.linear = nn.Linear(in_channels, latent_dim)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=4,
                                                             dim_feedforward=latent_dim * 2,
                                                             dropout=0.1)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer, num_layers=1)
        self.mu_token = nn.Parameter(torch.randn(latent_dim))
        self.logvar_token = nn.Parameter(torch.randn(latent_dim))


    def forward(self, input):
        x = self.linear(input)  # B T D
        B, T, D = input.shape

        lengths = [len(item) for item in input]

        mu_token = torch.tile(self.mu_token, (B,)).reshape(B, 1, -1)
        logvar_token = torch.tile(self.logvar_token, (B,)).reshape(B, 1, -1)

        x = torch.cat([mu_token, logvar_token, x], dim=1)

        x = x.permute(1, 0, 2)

        token_mask = torch.ones((B, 2), dtype=bool, device=input.get_device())
        mask = lengths_to_mask(lengths, input.get_device())

        aug_mask = torch.cat((token_mask, mask), 1)

        x = self.seqTransEncoder(x, src_key_padding_mask=~aug_mask)

        mu = x[0]
        logvar = x[1]
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        motion_sample = self.sample_from_distribution(dist).to(input.get_device())
        return motion_sample, dist

    def sample_from_distribution(self, distribution):
         return distribution.rsample()




class Decoder(nn.Module):
    def __init__(self,  output_3dmm_dim = 58, output_emotion_dim = 25, feature_dim = 128, device = 'cpu', max_seq_len=751, n_head = 4, window_size = 8, online = False):
        super(Decoder, self).__init__()

        self.feature_dim = feature_dim
        self.window_size = window_size
        self.device = device
        self.online = online

        self.vae_model = VAEModel(feature_dim, feature_dim)

        if self.online:
            self.lstm = nn.LSTM(feature_dim, feature_dim, 1 , batch_first = True)
            self.linear_3d = nn.Linear(output_3dmm_dim, feature_dim)
            self.linear_reaction = nn.Linear(feature_dim, feature_dim)
            decoder_layer_3d = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=4, dim_feedforward=2*feature_dim, batch_first=True)
            self.listener_reaction_decoder_3d = nn.TransformerDecoder(decoder_layer_3d, num_layers=1)



        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=n_head, dim_feedforward=2*feature_dim, batch_first=True)
        self.listener_reaction_decoder_1 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.listener_reaction_decoder_2 = nn.TransformerDecoder(decoder_layer, num_layers=1)


        self.biased_mask = init_biased_mask(n_head = n_head, max_seq_len = max_seq_len, period=max_seq_len)

        self.listener_reaction_3dmm_map_layer = nn.Linear(feature_dim, output_3dmm_dim)
        self.listener_reaction_emotion_map_layer = nn.Sequential(
            nn.Linear(feature_dim + output_3dmm_dim, feature_dim),
            nn.Linear(feature_dim, output_emotion_dim)
        )
        self.PE = PositionalEncoding(feature_dim)


    def forward(self, encoded_feature, past_reaction_3dmm = None, past_reaction_emotion = None):
        B, TS = encoded_feature.shape[0], encoded_feature.shape[1]
        if self.online:
            TL = self.window_size
        else:
            TL = TS
        motion_sample, dist = self.vae_model(encoded_feature)
        time_queries = torch.zeros(B, TL, self.feature_dim, device=encoded_feature.get_device())
        time_queries = self.PE(time_queries)
        tgt_mask = self.biased_mask[:, :TL, :TL].clone().detach().to(device=self.device).repeat(B,1,1)


        listener_reaction = self.listener_reaction_decoder_1(tgt=time_queries, memory=motion_sample.unsqueeze(1), tgt_mask=tgt_mask)
        listener_reaction = self.listener_reaction_decoder_2(listener_reaction, listener_reaction, tgt_mask=tgt_mask)



        if self.online and (past_reaction_3dmm is not None):
            past_reaction_3dmm = self.linear_3d(past_reaction_3dmm)
            past_reaction_3dmm_last = past_reaction_3dmm[:,-1]

            tgt_mask = self.biased_mask[:, :(TL + past_reaction_3dmm.shape[1]), :(TL + past_reaction_3dmm.shape[1])].detach().to(device=self.device).repeat(B,1,1)
            all_3dmm = torch.cat((past_reaction_3dmm, self.linear_reaction(listener_reaction)), dim = 1)
            listener_3dmm_out = self.listener_reaction_decoder_3d(all_3dmm, all_3dmm, tgt_mask=tgt_mask)
            frame_num = listener_3dmm_out.shape[1]
            listener_3dmm_out = listener_3dmm_out[:, (frame_num - TL):]

            listener_3dmm_out, _ = self.lstm(listener_3dmm_out, (past_reaction_3dmm_last.view(1, B, self.feature_dim).contiguous(), past_reaction_3dmm_last.view(1, B, self.feature_dim).contiguous()))


            listener_3dmm_out = self.listener_reaction_3dmm_map_layer(listener_3dmm_out)
        else:
            listener_3dmm_out = self.listener_reaction_3dmm_map_layer(listener_reaction)


        listener_emotion_out = self.listener_reaction_emotion_map_layer(
            torch.cat((listener_3dmm_out, listener_reaction), dim=-1))

        return listener_3dmm_out, listener_emotion_out, dist

    def reset_window_size(self, window_size):
        self.window_size = window_size


class SpeakerBehaviourEncoder(nn.Module):
    def __init__(self, img_size=224, audio_dim = 78, feature_dim = 128,  device = 'cpu'):
        super(SpeakerBehaviourEncoder, self).__init__()

        self.img_size = img_size
        self.audio_dim = audio_dim
        self.feature_dim = feature_dim
        self.device = device

        self.video_encoder = VideoEncoder(img_size=img_size, feature_dim=feature_dim, device=device)
        self.audio_feature_map = nn.Linear(self.audio_dim, self.feature_dim)
        self.fusion_layer = nn.Linear(self.feature_dim*2, self.feature_dim)


    def forward(self, video, audio):
        video_feature = self.video_encoder(video)
        audio_feature = self.audio_feature_map(audio)
        speaker_behaviour_feature = self.fusion_layer(torch.cat((video_feature, audio_feature), dim =-1))

        return  speaker_behaviour_feature



class TransformerVAE(nn.Module):
    def __init__(self, img_size=224, audio_dim = 78, output_3dmm_dim = 58, output_emotion_dim = 25, feature_dim = 128, seq_len=751, online = True, window_size = 8, device = 'cpu'):
        super(TransformerVAE, self).__init__()

        self.img_size = img_size
        self.feature_dim = feature_dim
        self.output_3dmm_dim = output_3dmm_dim
        self.output_emotion_dim = output_emotion_dim
        self.seq_len = seq_len
        self.online = online
        self.window_size = window_size

        self.speaker_behaviour_encoder = SpeakerBehaviourEncoder(img_size, audio_dim, feature_dim, device)
        self.reaction_decoder = Decoder(output_3dmm_dim = output_3dmm_dim, output_emotion_dim = output_emotion_dim, feature_dim = feature_dim,  device=device, window_size = self.window_size, online = online)
        self.fusion = nn.Linear(feature_dim + self.output_3dmm_dim + self.output_emotion_dim, feature_dim)

    def forward(self, speaker_video=None, speaker_audio=None, **kwargs):

        """
        input:
        video: (batch_size, seq_len, 3, img_size, img_size)
        audio: (batch_size, raw_wav)

        output:
        3dmm_vector: (batch_size, seq_len, output_3dmm_dim)
        emotion_vector: (batch_size, seq_len, output_emotion_dim)
        distribution: [dist_1,...,dist_n]
        """

        frame_num = speaker_video.shape[1]
        distribution = []
        if self.online:
            period = frame_num // self.window_size
            reaction_3dmm = torch.zeros((speaker_video.size(0), self.window_size, self.output_3dmm_dim), device = speaker_video.get_device())
            reaction_emotion = torch.zeros((speaker_video.size(0), self.window_size, self.output_emotion_dim), device = speaker_video.get_device())

            for i in range(0, period):
                speaker_video_, speaker_audio_ = speaker_video[:, : (i + 1) * self.window_size], speaker_audio[:, :  (i + 1) * self.window_size]
                encoded_feature = self.speaker_behaviour_encoder(speaker_video_, speaker_audio_)

                # modality fusion
                encoded_feature = self.fusion(torch.cat((encoded_feature, reaction_3dmm, reaction_emotion), dim = -1))

                if i != 0:
                    past_reaction_3dmm, past_reaction_emotion = reaction_3dmm[:, : i * self.window_size], reaction_emotion[:, : i * self.window_size]
                    current_reaction_3dmm, current_reaction_emotion = reaction_3dmm[:, i * self.window_size :], reaction_emotion[:, i * self.window_size :]
                    listener_3dmm_out, listener_emotion_out, dist = self.reaction_decoder(encoded_feature, past_reaction_3dmm)


                    reaction_3dmm = torch.cat((past_reaction_3dmm, listener_3dmm_out, current_reaction_3dmm), dim = 1)
                    reaction_emotion = torch.cat((past_reaction_emotion, listener_emotion_out, current_reaction_emotion), dim = 1)

                else:
                    listener_3dmm_out, listener_emotion_out, dist = self.reaction_decoder(encoded_feature)
                    reaction_3dmm =  torch.cat((listener_3dmm_out, reaction_3dmm), dim = 1)
                    reaction_emotion = torch.cat((listener_emotion_out, reaction_emotion), dim = 1)

                distribution.append(dist)

            listener_3dmm_out, listener_emotion_out = reaction_3dmm[:, :frame_num], reaction_emotion[:, :frame_num]

            return listener_3dmm_out, listener_emotion_out, distribution

        else:
            encoded_feature = self.speaker_behaviour_encoder(speaker_video, speaker_audio)
            listener_3dmm_out, listener_emotion_out, dist = self.reaction_decoder(encoded_feature)
            distribution.append(dist)
            return listener_3dmm_out, listener_emotion_out, distribution


    def reset_window_size(self, window_size):
        self.window_size = window_size
        self.reaction_decoder.reset_window_size(window_size)



if __name__ == "__main__":
    pass
