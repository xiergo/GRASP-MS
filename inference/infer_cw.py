import argparse
import os
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
        if '-ver-' in pdb_id:
            pdb_id0, ver = pdb_id.split('-ver-')
            restr=f'{pdb_id0}_{ver}'
        else:
            pdb_id0 = pdb_id
            restr = pdb_id0
        
        # pat_dict = super().get_pattern(pdb_id0)
        # restr_pat = f'{self.restr_dir}/{restr}.pkl'
        # pat_dict.update({'restr': restr_pat})
        pat_dict = {
            'raw_feat': f'{self.raw_feat_dir}/{pdb_id0}.pkl',
            'fasta': f'{self.fasta_dir}/{pdb_id0}.fasta',
            'restr': f'{self.restr_dir}/{restr}.pkl'
        }
        return pat_dict
        
    def get_data(self, pdb_id):
        
        raw_feature = self.get_feat(pdb_id)
        restr_file = self.get_files(pdb_id)['restr']
        with open(restr_file, 'rb') as f:
            restraints = pickle.load(f)
        return raw_feature, restraints
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Inputs for eval.py')
    parser.add_argument('--train_url', required=False, default="/job/output/", help='Location of training output')
    parser.add_argument('--data_url', required=False, default="/job/dataset/", help='Location of data')
    parser.add_argument('--data_config', default="/job/file/config/data-infer.yaml", help='data process config')
    parser.add_argument('--model_config', default="/job/file/config/model-infer.yaml", help='model config')
    parser.add_argument('--ckpt_dir', default="ft-grasp-v11-64", help='model config')
    parser.add_argument('--seq_len', default=1280, type=int)
    parser.add_argument('--mixed_precision', default=1, type=int)
    parser.add_argument('--multimer', default=1, type=int)
    parser.add_argument('--jobname', default='case_cw', type=str)
    parser.add_argument('--ckpt_ids', default=None, type=str)
    parser.add_argument('--baseline', default=0, type=int)
    
    arguments = parser.parse_args()
    print(arguments)
    
    # set path
    raw_feat_dir = f'{arguments.data_url}/case_cw/raw_feat'
    fasta_dir = f'{arguments.data_url}/case_cw/fasta'
    restr_dir = f'{arguments.data_url}/case_cw/restr'
    res_dir = f'{arguments.train_url}/grasp_infer_res/{arguments.ckpt_dir}_{arguments.jobname}'
    ckpt_dir = f'{arguments.train_url}/ckpt_dir/{arguments.ckpt_dir}'
    
    # set case
    # pdb_ids = ['9A07', '9A06_A4B-ver-closed', '9A06_A4B-ver-opennew']
    pdb_ids = ['9A07_new', '8CX0-ver-both', '8CX0-ver-interface', '9A07', '9A06_A4B-ver-closed', '9A06_A4B-ver-opennew']
    key = 'key1' # set key for monitering completion
    if arguments.ckpt_ids is not None:
        ckpt_ids = [int(i)*1000 for i in arguments.ckpt_ids.split('-')]
    else:
        ckpt_ids = [i*1000 for i in (8, 14, 20, 22)] + [22946222]
    ckpt_ids.sort()

    # set config
    data_gen = MyDataGenerator(raw_feat_dir, fasta_dir, restr_dir)
    model_gen = ModelGenerator(arguments, ckpt_dir)
    sn = infer_config(rotate_split=False, outdir=res_dir, key=key)
    sn.start_job()
    infer_batch(model_gen, data_gen, sn, pdb_ids, ckpt_ids, res_dir, baseline=bool(arguments.baseline), num_seed=5, check_tsv_exist=True)
    
    print(f'finish! {datetime.datetime.now()}')
    sn.complete()

    for i in range(3600):
        if sn.check_all_complete():
            break
        time.sleep(60)
        i += 1
        print(f'sleep {i} minute(s)')
        
        
        
