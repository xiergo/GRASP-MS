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
parser.add_argument('--output_dir', default='/job/output/test', help='Output directory for predictions')
parser.add_argument('--restr', default=None, required=False, help='Location of restraints pickle input, if not provided, will infer without restraints')
parser.add_argument('--ckpt_path', default="/job/output/ckpt_dir/ft-grasp-v11-64/step_8000.ckpt", help='ckpt path')
parser.add_argument('--seq_len', default=256, type=int) # sequence will be padded to this length
arguments = parser.parse_args()
print(arguments)
model_gen = ModelGenerator(arguments, arguments.ckpt_path)

with open(arguments.raw_feat, 'rb') as f:
    raw_feature = pickle.load(f)

df = grasp_infer(model_gen=model_gen, 
                 ckpt_id=8000,
                 raw_feature=raw_feature,
                 restraints=arguments.restr, 
                 output_prefix=f'{arguments.output_dir}/')

df.to_csv(f'{arguments.output_dir}/predictions.tsv', index=False, sep='\t')
print(df)
print("Inference done!")