import argparse
import os
import glob
import time
import datetime
import pickle
import pandas as pd
import numpy as np
from restraint_sample import BINS
from utils_infer import infer_config, infer_batch, DataGenerator, ModelGenerator, grasp_infer

parser = argparse.ArgumentParser(description='Inputs for eval.py')
parser.add_argument('--raw_feat', default='/job/dataset/csp/raw_feat/5JDS.pkl', help='Location of raw features pickle input')
parser.add_argument('--output_dir', default='/job/output', help='Output directory for predictions')
# parser.add_argument('--restr', default=None, required=False, help='Location of restraints pickle input, if not provided, will infer without restraints')
parser.add_argument('--ckpt_path', default="/job/output/ckpt_dir/ft-grasp-v11-64/step_8000.ckpt", help='ckpt path')
parser.add_argument('--data_config', default="/job/file/config/data-infer.yaml", help='data process config')
parser.add_argument('--model_config', default="/job/file/config/model-infer.yaml", help='model config')
parser.add_argument('--seq_len', default=256, type=int) # sequence will be padded to this length
parser.add_argument('--mixed_precision', default=1, type=int)
parser.add_argument('--multimer', default=1, type=int)


arguments = parser.parse_args()
print(arguments)
model_gen = ModelGenerator(arguments, arguments.ckpt_path)

with open(arguments.raw_feat, 'rb') as f:
    raw_feature = pickle.load(f)

s = raw_feature['msa'].shape[1]
if s>arguments.seq_len:
    for k, v in raw_feature.items():
        v_new = v.copy()
        for i in range(len(v.shape)):
            if v.shape[i]==s:
                v_new = np.take(v_new, range(arguments.seq_len), axis=i)
        raw_feature[k] = v_new

for i in [1,2,3,4,5]:
    print(f"Inference for recycle {i}", datetime.datetime.now())
    output_prefix = f'{arguments.output_dir}/grasp_speed1/seq_len_{arguments.seq_len}/recycle_{i}'
    df = grasp_infer(model_gen=model_gen, 
                    ckpt_id=8000,
                    raw_feature=raw_feature,
                    restraints=None, 
                    output_prefix=output_prefix, 
                    iter=1,
                    seed=0,
                    num_recycle=i,)

    print(df.to_string())
    df.to_csv(f'{output_prefix}_info.tsv', index=False, sep='\t')
    print("Inference done for recycle", i, datetime.datetime.now())