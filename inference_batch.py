import argparse
import os
import re
import pickle
import glob
import time
import datetime
import numpy as np
from restraint_sample import BINS
from utils_infer import infer_config, infer_batch, DataGenerator, ModelGenerator
from mindsponge1.common.residue_constants import restypes
from Bio.SeqUtils import seq1
 

class MyDataGenerator(DataGenerator):
    def __init__(self, raw_feat_dir, restr_dir):
        super().__init__(raw_feat_dir)
        self.restr_dir = restr_dir
    
    def get_data(self, pdb_id):
        raw_feature = self.get_feat(pdb_id)  
        restr_file = f'{self.restr_dir}/{pdb_id}.pkl'
        with open(restr_file, 'rb') as f:
            restraints = pickle.load(f)
        restraints['asym_id'] = raw_feature['asym_id']
        return raw_feature, restraints
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Inputs for eval.py')
    parser.add_argument('--train_url', required=False, default="/home/ccs_cli/xieyh/output", help='Location of training output')
    parser.add_argument('--data_url', required=False, default="/home/ccs_cli/xieyh/dataset", help='Location of data')
    parser.add_argument('--data_config', default="./config/data-infer.yaml", help='data process config')
    parser.add_argument('--model_config', default="./config/model-infer.yaml", help='model config')
    parser.add_argument('--ckpt_dir', default="ft-grasp-v6-notfix-nohard-32", help='model config')
    parser.add_argument('--seq_len', default=2048, type=int)
    parser.add_argument('--mixed_precision', default=1, type=int)
    parser.add_argument('--multimer', default=1, type=int)
    parser.add_argument('--uniform', default=1, type=int)
    parser.add_argument('--jobname', default='xl_high', type=str)
    
    
    arguments = parser.parse_args()
    print(arguments)
    
    # set path
    raw_feat_dir = f'{arguments.data_url}/xl_high/features'
    restr_dir = f'{arguments.data_url}/xl_high/'
    res_dir = f'{arguments.train_url}/grasp_infer_res/{arguments.ckpt_dir}_{arguments.jobname}_{arguments.uniform}'
    ckpt_dir = f'{arguments.train_url}/ckpt_dir/{arguments.ckpt_dir}'
    
    data_gen = MyDataGenerator(raw_feat_dir, restr_dir)
    model_gen = ModelGenerator(arguments, ckpt_dir)
    sn = infer_config(rotate_split=False)
    
    pdb_ids = [i.split('.')[0] for i in os.listdir(raw_feat_dir)]
    ckpt_ids = [i for i in [8000, 14000, 20000, 22000, 21946222] if os.path.isfile(f'{ckpt_dir}/step_{i}.ckpt')]
    infer_batch(model_gen, data_gen, sn, pdb_ids, ckpt_ids, res_dir, baseline=False)
    
    print(f'finish! {datetime.datetime.now()}')
    
    # for i in range(120):
    #     time.sleep(60)
    #     i += 1
    #     print(f'sleep {i} minute(s)')
        
        
        
