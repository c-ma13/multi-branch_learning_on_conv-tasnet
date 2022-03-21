#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute SI-SDR, PESQ as the evaluation metrics
"""

import argparse
import json
import os
from pathlib import Path
import soundfile as sf


import torch
import numpy as np
from tqdm import tqdm
from pypesq import pesq as calc_pesq
from pystoi.stoi import stoi as calc_stoi

from nnet.conv_tasnet_student import TasNet as nnet
from data.eval_data import all_loaders_in_one

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

eval_fs = 8000
save_file = 1
use_cuda = True
exp_name = "student"
exp_path = Path("/data/machao/multi_task_conv_tasnet_whamr/exp/") / exp_name

parser = argparse.ArgumentParser()
parser.add_argument('--exp', default=exp_path, type=Path, help='exp name')
args = parser.parse_args()


def dump_json(obj, fdir, name):
    """
    Dump python object in json
    """
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    with open(os.path.join(fdir, name), "w") as f:
        json.dump(obj, f, indent=4, sort_keys=False)


def calc_sisdr(sig_cln, sig_proc, eps=1e-8):
    '''
    calculate Si-SDR for one mono audio
    '''
    def _norm(x):
        return np.sum(x ** 2)
    sig_proc = np.asarray(sig_proc, dtype=np.float32)
    sig_cln = np.asarray(sig_cln, dtype=np.float32)
    if np.max(sig_cln) > 2:
        sig_cln /= (2**15)
        sig_proc /= (2**15)
    #  sig_cln = sig_cln - np.mean(sig_cln)
    #  sig_proc = sig_proc - np.mean(sig_proc)
    sig_tar = np.sum(sig_cln * sig_proc) * sig_cln / (_norm(sig_cln) + eps)
    upp = _norm(sig_tar)
    low = _norm(sig_proc - sig_tar)
    return 10 * np.log10(upp) - 10 * np.log10(low)


class MetricsCalc(object):
    def __init__(self, save_path, fs):
        self.save_path = save_path
        self.data = {}
        self.fs = fs

    def add(self, key, sep, ref, mix):
        metrics = {}

        try:
            metrics["sisdr"] = calc_sisdr(ref, sep)
            metrics["pesq"] = calc_pesq(ref, sep, self.fs)
            metrics["stoi"] = calc_stoi(ref, sep, self.fs)

            metrics["sisdri"] = metrics["sisdr"] - calc_sisdr(ref, mix)
            metrics["pesqi"] = metrics["pesq"] - calc_pesq(ref, mix, self.fs)
            metrics["stoii"] = metrics["stoi"] - calc_stoi(ref, mix, self.fs)
            self.data[key] = metrics
            print(metrics)
        except:
            print("processing error. key is ", key)

    def summary(self):
        N = len(self.data)
        sisdrs, pesqs, stois, sisdris, pesqis, stoiis = [], [], [], [], [], []
        for key, metrics in self.data.items():
            sisdrs.append(metrics["sisdr"])
            pesqs.append(metrics["pesq"])
            stois.append(metrics["stoi"])
            sisdris.append(metrics["sisdri"])
            pesqis.append(metrics["pesqi"])
            stoiis.append(metrics["stoii"])
        self.metrics = {"sisdr": np.nanmean(sisdrs),
                        "pesq": np.nanmean(pesqs),
                        "stoi": np.nanmean(stois),
                        "sisdri": np.nanmean(sisdris),
                        "pesqi": np.nanmean(pesqis),
                        "stoii": np.nanmean(stoiis)}
        print("metrics on test dataset:")
        print(self.metrics)
        dump_json(self.data, self.save_path, "eval_data.json")
        dump_json(self.metrics, self.save_path, "eval_out.json")


def run():
    save_path = args.exp / "evaluation_results"
    out_path = args.exp / "test_wavs"
    fs = eval_fs
    calcer = MetricsCalc(save_path, fs)

    cpt = torch.load(args.exp / "best.pth.tar")
    print("loadding model from {} at {} epochs".format(args.exp, cpt["epoch"]))
    model_info = cpt["model_info"]
    model = nnet(model_info)
    model_state_dict = cpt["model_state_dict"]
    model.load_state_dict(model_state_dict)
    if use_cuda:
        model.cuda()

    infer_data_loader = all_loaders_in_one(1)
    with torch.no_grad():
        for utt, egs in enumerate(tqdm(infer_data_loader)):
            if utt == 2343:
                continue
            if use_cuda:
                egs = list(map(lambda x: x.cuda(), egs))
            infer_mix, infer_s1, infer_s2 = egs

            infer_est, _, _ = model(infer_mix)
            infer_est_s1 = infer_est[:, 0, :]
            infer_est_s2 = infer_est[:, 1, :]

            if use_cuda:
                infer_mix = infer_mix.cpu()
                infer_s1 = infer_s1.cpu()
                infer_s2 = infer_s2.cpu()
                infer_est_s1 = infer_est_s1.cpu()
                infer_est_s2 = infer_est_s2.cpu()

            mix = infer_mix[0].numpy()
            s1 = infer_s1[0].numpy()
            s2 = infer_s2[0].numpy()
            est1 = infer_est_s1[0].numpy()
            est2 = infer_est_s2[0].numpy()

            if save_file == 1 and utt < 10:
                tmp_out_path = out_path / str(utt)
                tmp_out_path.mkdir(parents=True, exist_ok=True)
                sf.write(tmp_out_path / "mix.wav", mix, fs, subtype="FLOAT")
                sf.write(tmp_out_path / "s1.wav", s1, fs, subtype="FLOAT")
                sf.write(tmp_out_path / "est1.wav", est1, fs, subtype="FLOAT")
                sf.write(tmp_out_path / "s2.wav", s2, fs, subtype="FLOAT")
                sf.write(tmp_out_path / "est2.wav", est2, fs, subtype="FLOAT")

            tmp_sisdr1 = calc_sisdr(s1, est1) + calc_sisdr(s2, est2)
            tmp_sisdr2 = calc_sisdr(s2, est1) + calc_sisdr(s1, est2)
            if tmp_sisdr1 > tmp_sisdr2:
                calcer.add(2*utt, est1, s1, mix)
                calcer.add(2*utt+1, est2, s2, mix)
            else:
                calcer.add(2*utt, est1, s2, mix)
                calcer.add(2*utt+1, est2, s1, mix)
        calcer.summary()


if __name__ == "__main__":
    run()
