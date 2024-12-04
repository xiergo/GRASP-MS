import argparse
import os
import glob
import pickle
import time
import datetime
import pandas as pd
import numpy as np
from restraint_sample import BINS
from utils_infer import infer_config, infer_batch, DataGenerator, ModelGenerator


class MyDataGenerator(DataGenerator):
    def __init__(self, raw_feat_dir, fasta_dir, restr_dir, reorder=False):
        super().__init__(raw_feat_dir, fasta_dir, reorder)
        self.restr_dir = restr_dir

    def get_pattern(self, pdb_id):
        pdb_id0, restr_type = pdb_id.split('-')
        pat_dict = super().get_pattern(pdb_id0)
        restr_pat = f'{self.restr_dir}/{pdb_id0}-{restr_type}*pkl'
        pat_dict.update({'restr': restr_pat})
        return pat_dict
        
    def get_data(self, pdb_id):
        raw_feature = self.get_feat(pdb_id)
        restr_file = self.get_files(pdb_id)['restr']
        with open(restr_file, 'rb') as f:
            restraints = pickle.load(f)
        restraints['asym_id'] = raw_feature['asym_id']
        return raw_feature, restraints
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Inputs for eval.py')
    parser.add_argument('--train_url', required=False, default="/job/output/", help='Location of training output')
    parser.add_argument('--data_url', required=False, default="/job/dataset/", help='Location of data')
    parser.add_argument('--data_config', default="/job/file/config/data-infer.yaml", help='data process config')
    parser.add_argument('--model_config', default="/job/file/config/model-infer.yaml", help='model config')
    parser.add_argument('--ckpt_dir', default="ft-grasp-v11-64", help='model config')
    parser.add_argument('--seq_len', default=1024, type=int)
    parser.add_argument('--mixed_precision', default=1, type=int)
    parser.add_argument('--multimer', default=1, type=int)
    parser.add_argument('--jobname', default='debug', type=str)
    parser.add_argument('--ckpt_ids', default=None, type=str)
    parser.add_argument('--baseline', default=0, type=int)
    
    arguments = parser.parse_args()
    print(arguments)
    
    # set path
    raw_feat_dir = f'{arguments.data_url}/1GZM/feat'
    fasta_dir = f'{arguments.data_url}/1GZM/fasta'
    restr_dir = f'{arguments.data_url}/1GZM/restr'
    res_dir = f'{arguments.train_url}/grasp_infer_res/{arguments.ckpt_dir}_{arguments.jobname}'
    ckpt_dir = f'{arguments.train_url}/ckpt_dir/{arguments.ckpt_dir}'
    
    data_gen = MyDataGenerator(raw_feat_dir, fasta_dir, restr_dir)
    model_gen = ModelGenerator(arguments, ckpt_dir)
    sn = infer_config(rotate_split=False, outdir=res_dir)
    sn.start_job()
    
    pdb_ids = [f'1GZM-{i}' for i in ['all', 'part']]
    if arguments.ckpt_ids is not None:
        ckpt_ids = [int(i)*1000 for i in arguments.ckpt_ids.split('-')]
    else:
        ckpt_ids = [i*1000 for i in [8, 14, 20, 22]]
    ckpt_ids.sort()
    
    infer_batch(model_gen, data_gen, sn, pdb_ids, ckpt_ids, res_dir, baseline=bool(arguments.baseline), num_seed=5, check_tsv_exist=True)
    
    print(f'finish! {datetime.datetime.now()}')
    sn.complete()

    for i in range(3600):
        if sn.check_all_complete():
            break
        time.sleep(60)
        i += 1
        print(f'sleep {i} minute(s)')
        
        
        
