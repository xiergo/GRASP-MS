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

# with open(arguments.raw_feat, 'rb') as f:
#     raw_feature = pickle.load(f)

# df = grasp_infer(model_gen=model_gen, 
#                  ckpt_id=8000,
#                  raw_feature=raw_feature,
#                  restraints=arguments.restr, 
#                  output_prefix=f'{arguments.output_dir}/{name}')


sn = infer_config(rotate_split=False, outdir=arguments.output_dir)
sn.start_job()

pdb_ids = ['muor_single-restraints_26', 'muor_single-restraints_36']
if arguments.ckpt_ids is not None:
    ckpt_ids = [int(i)*1000 for i in arguments.ckpt_ids.split('-')]
else:
    ckpt_ids = [i*1000 for i in (8, 14, 20, 22)] + [22946222]
ckpt_ids.sort()


class MyDataGenerator(DataGenerator):
    def __init__(self, raw_feat_dir):
        super().__init__(raw_feat_dir)

    def get_data(self, case_id):
        pdb_id, restr_id = case_id.split('-')
        raw_feature = self.get_feat(pdb_id)
        restr_file = f'{self.raw_feat_dir}/{restr_id}.pkl'
        with open(restr_file, 'rb') as f:
            restraints = pickle.load(f)
        restraints['asym_id'] = raw_feature['asym_id']
        return raw_feature, restraints
data_gen = MyDataGenerator('/job/dataset/muor')

infer_batch(model_gen, data_gen, sn, pdb_ids, ckpt_ids, arguments.output_dir)

print(f'finish! {datetime.datetime.now()}')
sn.complete()

print(df)
print("Inference done!")