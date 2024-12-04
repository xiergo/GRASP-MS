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
parser.add_argument('--train_url', required=False, default="/job/output/", help='Location of training output')
parser.add_argument('--data_url', required=False, default="/job/dataset/", help='Location of data')
parser.add_argument('--data_config', default="/job/file/config/data-infer.yaml", help='data process config')
parser.add_argument('--model_config', default="/job/file/config/model-infer.yaml", help='model config')
parser.add_argument('--ckpt_dir', default="ft-grasp-v11-64", help='model config')
parser.add_argument('--seq_len', default=384, type=int)
parser.add_argument('--mixed_precision', default=1, type=int)
parser.add_argument('--multimer', default=1, type=int)
parser.add_argument('--jobname', default='muor', type=str)
parser.add_argument('--ckpt_ids', default=None, type=str)
parser.add_argument('--baseline', default=0, type=int)
arguments = parser.parse_args()
print(arguments)
res_dir = f'{arguments.train_url}/grasp_infer_res/{arguments.ckpt_dir}_{arguments.jobname}'
ckpt_dir = f'{arguments.train_url}/ckpt_dir/{arguments.ckpt_dir}'
sn = infer_config(rotate_split=False, outdir=res_dir)
sn.start_job()
model_gen = ModelGenerator(arguments, ckpt_dir)

pdb_ids = ['muor_single-restraints_26', 'muor_single-restraints_36']
if arguments.ckpt_ids is not None:
    ckpt_ids = [int(i)*1000 for i in arguments.ckpt_ids.split('-')]
else:
    ckpt_ids = [i*1000 for i in (8, 14, 20, 22)] + [22946222]
ckpt_ids.sort()


class MyDataGenerator(DataGenerator):
    def __init__(self, raw_feat_dir):
        super().__init__(raw_feat_dir)
    def get_pattern(self, pdb_id):
        pdb_id, restr_id = pdb_id.split('-')
        pat_dict = {
            'raw_feat': f'{self.raw_feat_dir}/{pdb_id}*.pkl',
            'restr': f'{self.raw_feat_dir}/{restr_id}*.pkl'
        }
        return pat_dict
    def get_data(self, pdb_id):
        raw_feature = self.get_feat(pdb_id)
        restr_file = self.get_files(pdb_id)['restr']
        with open(restr_file, 'rb') as f:
            restraints = pickle.load(f)
        restraints['asym_id'] = raw_feature['asym_id']
        return raw_feature, restraints
data_gen = MyDataGenerator('/job/dataset/muor')

infer_batch(model_gen, data_gen, sn, pdb_ids, ckpt_ids, res_dir)

print(f'finish! {datetime.datetime.now()}')
sn.complete()

print("Inference done!")