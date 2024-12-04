import argparse
import os
# import re
import glob
import pickle
import time
import datetime
# import pandas as pd
import numpy as np
# from restraint_sample import BINS
from utils_infer import infer_config, infer_batch, DataGenerator, ModelGenerator
# from exp_info_sample_benchmark import generate_interface_and_restraints


class MyDataGenerator(DataGenerator):
    def __init__(self, raw_feat_dir, restr_dir):
        super().__init__(raw_feat_dir)
        self.restr_dir = restr_dir

    def get_pattern(self, pdb_id):
        # 7T4E_dimer_A_C_interface_10
        pat_dict = {
            'raw_feat': f'{self.raw_feat_dir}/{pdb_id}.pkl',
            'restr': f'{self.restr_dir}/{pdb_id}_restraints.pkl'
        }
        return pat_dict
    
    def get_data(self, pdb_id):
        
        raw_feature = self.get_feat(pdb_id)  
        restr_file = self.get_files(pdb_id)['restr']
        with open(restr_file, 'rb') as f:
            restraints = pickle.load(f)
        assert np.abs(restraints['asym_id'] - raw_feature['asym_id']).sum() == 0, np.abs(restraints['asym_id'] - raw_feature['asym_id']).sum()
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
    parser.add_argument('--jobname', default='simxlv2_5perc', type=str)
    parser.add_argument('--ckpt_ids', default=None, type=str)
    
    arguments = parser.parse_args()
    print(arguments)
    
    # set path
    raw_feat_dir = f'{arguments.data_url}/xl_simulated_v2/raw_feat'
    restr_dir = f'{arguments.data_url}/xl_simulated_v2/out2_xls_restr_5perc'
    res_dir = f'{arguments.train_url}/grasp_infer_res/{arguments.ckpt_dir}_{arguments.jobname}'
    ckpt_dir = f'{arguments.train_url}/ckpt_dir/{arguments.ckpt_dir}'
    
    data_gen = MyDataGenerator(raw_feat_dir, restr_dir)
    model_gen = ModelGenerator(arguments, ckpt_dir)
    sn = infer_config(rotate_split=False, outdir=res_dir)
    sn.start_job()
    # /dl/atlas_dls/1/24/user_data/dataset/multimer_contact/basic_benchmark_dataset/out5_sampled_restr/interface/7T4E_dimer_A_C_interface_10_raw.pkl
    restrs = glob.glob(f'{restr_dir}/*.pkl')
    pdb_ids = [os.path.basename(i).replace('_restraints.pkl', '') for i in restrs]
    pdb_ids.sort()
    print(len(pdb_ids), pdb_ids[:10])

    if arguments.ckpt_ids is not None:
        ckpt_ids = [int(i)*1000 for i in arguments.ckpt_ids.split('-')]
    else:
        ckpt_ids = [i*1000 for i in [8, 14, 20, 22]] + [22946222]
        # ckpt_ids = [21946222,]
    ckpt_ids.sort()
    print('ckpt_ids', ckpt_ids)

    infer_batch(model_gen, data_gen, sn, pdb_ids, ckpt_ids, res_dir, baseline=False, iter=5, num_seed=5, check_tsv_exist=True)
    
    print(f'finish! {datetime.datetime.now()}')
    sn.complete()

    for i in range(3600):
        if sn.check_all_complete():
            break
        time.sleep(60)
        i += 1
        print(f'sleep {i} minute(s)')
        
        
        
