import torch
import torch.nn as nn
from utils.compute_distance_fun import compute_distance


NUM_SAMPLE=10
class MHP(nn.Module):
    def __init__(self, p=None, c=None, m=None, no_inverse=False, dist="MSE", neighbor_pattern='nearest'):
        super().__init__()
        self.perceptual_processor = p
        self.cognitive_processor = c
        self.motor_processor = m
        self.cal_dist = {
            "DTW": compute_distance,
            "MSE": nn.functional.mse_loss,}

        self.no_inverse = no_inverse
        self.neighbor_pattern = neighbor_pattern

    def forward_features(self, video_inputs, audio_inputs):
        if video_inputs is None:
            fused_features = audio_inputs
        elif audio_inputs is None:
            fused_features = video_inputs
        else:
            # B, T, 64
            fused_features = self.perceptual_processor(video_inputs, audio_inputs)
        # cog_outputs: 4, 25, 50 ---> B, N, T
        # edge: [4, 8, 25, 25] it does not change with T
        # params: 0.0?
        cog_outputs, edge, params = self.cognitive_processor(fused_features)
        return cog_outputs, edge, params

    def get_nearest(self, features, edge, targets):
        # B, 750, 25
        with torch.no_grad():
            if not self.no_inverse:
                self.motor_processor.eval()
                predictions = self.motor_processor.inverse(features, edge=edge)
            else:
                predictions = features
            B = predictions.shape[0]
            nearest_idx = []
            for i in range(B):
                if len(targets[1]) == None:
                    nearest_idx.append(0)
                    continue

                pred = predictions[i].unsqueeze(0)
                pair_targets = targets[i]
                min_dist = None
                for i, pair_target in enumerate(pair_targets):
                    if pair_target == None:
                        continue
                    dist = self.cal_dist(pred, pair_target.transpose(1, 0).unsqueeze(0))
                    if min_dist == None or dist < min_dist:
                        min_dist = dist
                        min_inx = i

                nearest_idx.append(min_inx)

        nearest_targets = [targets[i][idx].transpose(1, 0) for i, idx in enumerate(nearest_idx)]
        nearest_targets = torch.stack(nearest_targets, dim=0)
        return nearest_targets

    def forward(self, video_inputs, audio_inputs, targets, lengthes=None):
        # print('--------------------  Forward  --------------------')
        # speaker_feature: 4, 25, 50
        speaker_feature, edge, params = self.forward_features(video_inputs, audio_inputs)
        if not self.no_inverse:
            if self.neighbor_pattern == 'nearest':
                targets = self.get_nearest(speaker_feature, edge, targets)
                targets.requires_grad = True
            elif self.neighbor_pattern == 'all':
                edge = torch.repeat_interleave(edge, repeats=torch.tensor(lengthes, device=edge.device), dim=0)
            else:
                targets = targets

            self.motor_processor.train()
            # Encode all appropriate real facial reactions to a GMGD distribution
            # listener_feature: B, N, D
            listener_feature, logdets = self.motor_processor(targets, edge)

            return speaker_feature, listener_feature, params, edge, targets, logdets

        else:
            # Decode samples to listener appropriate facial reactions
            listerer_feature, logdets = self.motor_processor(speaker_feature, edge)
            nearest_targets = targets

            return listerer_feature, nearest_targets, logdets

    def inverse(self, video_inputs, audio_inputs, cal_norm, threshold=None):
        speaker_feature, edge, params = self.forward_features(video_inputs, audio_inputs)
        if not self.no_inverse:
            speaker_feature = self.sample(speaker_feature, threshold)
            predictions = self.motor_processor.inverse(speaker_feature, edge=edge, cal_norm=cal_norm)
        else:
            speaker_feature = self.sample(speaker_feature, threshold)
            predictions, _ = self.motor_processor(speaker_feature, edge)
        return predictions.transpose(2, 1)

    def onlyInverseMotor(self, features, edge):
        print('='*50)
        print('--------------------Only Inverse--------------------')
        with torch.no_grad():
            self.motor_processor.eval()
            outputs = self.motor_processor.inverse(features, edge=edge)

        return outputs

    def sample(self, speaker_feature, threshold=None):
        noise = torch.randn(speaker_feature.shape, device=speaker_feature.device)
        if threshold is None:
            return speaker_feature + noise

        threshold = torch.sqrt(torch.tensor([threshold], device=speaker_feature.device))

        max_abs = torch.max(torch.abs(noise))
        if max_abs <= threshold:
            return speaker_feature + noise
        scale = threshold / max_abs
        scaled_noise = noise * scale
        return speaker_feature + scaled_noise

