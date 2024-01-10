import os
import torch
import argparse
import numpy as np
import pandas as pd

from metric import *

SAMPLE_NUMS = 10

def parse_arg():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Param
    parser.add_argument('--data-dir', default="../data/react_clean", type=str, help="dataset path")
    parser.add_argument('--pred-dir', default="../data/react_clean/outputs/Gmm-logs", type=str, help="the path of saved predictions")
    parser.add_argument('--split', type=str, help="split of dataset", choices=["val", "test"], required=True)
    parser.add_argument('--threads', default=32, type=int, help="num max of threads")
    
    args = parser.parse_args()
    return args

def replace_pid(emotion_path):
    if 'NoXI' in emotion_path:
        emotion_path=emotion_path.replace('Novice_video','P2')
        emotion_path=emotion_path.replace('Expert_video','P1')

    if 'Emotion/RECOLA/group' in emotion_path:
        emotion_path=emotion_path.replace('P25','P1')
        emotion_path=emotion_path.replace('P26','P2')
        emotion_path=emotion_path.replace('P41','P1')
        emotion_path=emotion_path.replace('P42','P2')
        emotion_path=emotion_path.replace('P45','P1')
        emotion_path=emotion_path.replace('P46','P2')

    return emotion_path

def main(args):
    _list_path = pd.read_csv(os.path.join(args.data_dir, args.split + '.csv'), header=None, delimiter=',')
    _list_path = _list_path.drop(0)

    speaker_path_list = [path for path in list(_list_path.values[:, 1])] + [path for path in list(_list_path.values[:, 2])]
    listener_path_list = [path for path in list(_list_path.values[:, 2])] + [path for path in list(_list_path.values[:, 1])]

    listener_emotion_gt_list = []
    listener_emotion_pred_list = []
    speaker_emotion_list = []

    for index in range(len(speaker_path_list)):
        # speaker emotion
        speaker_path = speaker_path_list[index]
        speaker_emotion_path = os.path.join(args.data_dir, args.split, 'Emotion', speaker_path+'.csv')
        
        speaker_emotion_path = replace_pid(speaker_emotion_path)
        speaker_emotion = pd.read_csv(speaker_emotion_path, header=None, delimiter=',')
        speaker_emotion = torch.from_numpy(np.array(speaker_emotion.drop(0)).astype(np.float32))

        speaker_emotion_list.append(speaker_emotion)

        # listener emotion
        listener_path = listener_path_list[index]
        listener_emotion_path = os.path.join(args.data_dir, args.split, 'Emotion', listener_path+'.csv')
        
        listener_emotion_path = replace_pid(listener_emotion_path)
        listener_emotion = pd.read_csv(listener_emotion_path, header=None, delimiter=',')
        listener_emotion = torch.from_numpy(np.array(listener_emotion.drop(0)).astype(np.float32))

        listener_emotion_gt_list.append(listener_emotion)

        # predicted listener's emotions
        listener_emotion_pred = []
        for j in range(SAMPLE_NUMS):
            pred_path = os.path.join(args.pred_dir, args.split, speaker_path, 'result-' + str(j) + '.pth')
            listener_emotion_pred.append(torch.load(pred_path).cpu())

        listener_emotion_pred = torch.stack(listener_emotion_pred)

        listener_emotion_pred_list.append(listener_emotion_pred)

    speaker_emotion_gt = torch.stack(speaker_emotion_list, dim = 0)
    listener_emotion_gt = torch.stack(listener_emotion_gt_list, dim = 0)
    all_listener_emotion_pred = torch.stack(listener_emotion_pred_list)

    print("-----------------Evaluating Metric-----------------")

    p = args.threads

    # If you have problems running function compute_TLCC_mp, please replace this function with function compute_TLCC
    TLCC = compute_TLCC_mp(all_listener_emotion_pred, speaker_emotion_gt, p=p)

    # If you have problems running function compute_FRC_mp, please replace this function with function compute_FRC
    FRC = compute_FRC_mp(args.data_dir, all_listener_emotion_pred, listener_emotion_gt, val_test=args.split, p=p)

    # If you have problems running function compute_FRD_mp, please replace this function with function compute_FRD
    FRD = compute_FRD_mp(args.data_dir, all_listener_emotion_pred, listener_emotion_gt, val_test=args.split, p=p)

    FRDvs = compute_FRDvs(all_listener_emotion_pred)
    FRVar  = compute_FRVar(all_listener_emotion_pred)
    smse  = compute_s_mse(all_listener_emotion_pred)

    print("Metric: | FRC: {:.5f} | FRD: {:.5f} | S-MSE: {:.5f} | FRVar: {:.5f} | FRDvs: {:.5f} | TLCC: {:.5f}".format(FRC, FRD, smse, FRVar, FRDvs, TLCC))
    print("Latex-friendly --> model_name & {:.2f} & {:.2f} & {:.4f} & {:.4f} & {:.4f} & - & {:.2f} \\\\".format( FRC, FRD, smse, FRVar, FRDvs, TLCC))

if __name__=="__main__":
    args = parse_arg()
    os.environ["NUMEXPR_MAX_THREADS"] = '32'
    main(args)