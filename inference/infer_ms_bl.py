import argparse
import os
import re
import glob
import pickle
import time
import datetime
import pandas as pd
import numpy as np
from restraint_sample import BINS
from utils_infer import infer_config, infer_batch, DataGenerator, ModelGenerator
from exp_info_sample_benchmark import generate_interface_and_restraints



def generate_restraints(contact_feat, inter_num, intra_num, interface_num):
    
    _, sbr_mask, interface_mask, _, _, _ = \
            generate_interface_and_restraints(contact_feat, training=False,
                                                num_inter=inter_num,
                                                num_intra=intra_num,
                                                num_interface=interface_num,
                                                only_contact_sbr=True,
                                                seed=20230820)
    
    seq_len = sbr_mask.shape[0]
    sbr = np.zeros((seq_len, seq_len, len(BINS)+1))
    contact = np.zeros(len(BINS)+1)
    contact[:int((BINS<=8.0).sum())] = 1
    contact /= contact.sum()
    for i, j in zip(*np.where(sbr_mask>=0.5)):
        sbr[i, j] = contact
    
    restraint = {'sbr': sbr, 'sbr_mask': sbr_mask, 'interface_mask': interface_mask}
    return restraint


class MyDataGenerator(DataGenerator):
    def __init__(self, raw_feat_dir):
        super().__init__(raw_feat_dir)
        # self.restr_dir = restr_dir

    def get_pattern(self, pdb_id):
        # 7T4E_dimer_A_C_interface_10
        # pdb_id0, chain_pair, ty, num = re.search(r'(\w+)_dimer_(\w+_\w+)_(\w+)_(\d+)', pdb_id).groups()
        pat_dict = {
            'raw_feat': f'{self.raw_feat_dir}/{pdb_id}.pkl',
            # 'restr': f'{self.restr_dir}/{ty}/{pdb_id}_raw.pkl'
        }
        return pat_dict
    
    def get_data(self, pdb_id):
        
        raw_feature = self.get_feat(pdb_id)  
        # restr_file = self.get_files(pdb_id)['restr']
        # with open(restr_file, 'rb') as f:
        #     restraints = pickle.load(f)
        # assert np.abs(restraints['asym_id'] - raw_feature['asym_id']).sum() == 0, np.abs(restraints['asym_id'] - raw_feature['asym_id']).sum()
        restraints=None
        return raw_feature, restraints
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Inputs for eval.py')
    parser.add_argument('--train_url', required=False, default="/job/output/", help='Location of training output')
    parser.add_argument('--data_url', required=False, default="/job/dataset/", help='Location of data')
    parser.add_argument('--data_config', default="/job/file/config/data-infer.yaml", help='data process config')
    parser.add_argument('--model_config', default="/job/file/config/model-infer.yaml", help='model config')
    parser.add_argument('--ckpt_dir', default="ft-grasp-v13-64", help='model config') #46200
    parser.add_argument('--seq_len', default=1280, type=int)
    parser.add_argument('--mixed_precision', default=1, type=int)
    parser.add_argument('--multimer', default=1, type=int)
    parser.add_argument('--jobname', default='ms_bl', type=str)
    parser.add_argument('--ckpt_ids', default=None, type=str)
    
    arguments = parser.parse_args()
    print(arguments)
    
    # set path
    raw_feat_dir = f'{arguments.data_url}/ms_baseline_dataset/raw_feat'
    # restr_dir = f'{arguments.data_url}/basic_benchmark_dataset/out5_sampled_restr'
    res_dir = f'{arguments.train_url}/grasp_infer_res/{arguments.ckpt_dir}_{arguments.jobname}'
    ckpt_dir = f'{arguments.train_url}/ckpt_dir/{arguments.ckpt_dir}'
    
    data_gen = MyDataGenerator(raw_feat_dir)#, restr_dir)
    model_gen = ModelGenerator(arguments, ckpt_dir)
    sn = infer_config(rotate_split=False, outdir=res_dir)
    sn.start_job()
    # /dl/atlas_dls/1/24/user_data/dataset/multimer_contact/basic_benchmark_dataset/out5_sampled_restr/interface/7T4E_dimer_A_C_interface_10_raw.pkl
    # restrs = glob.glob(f'{restr_dir}/*/*_raw.pkl')

    # df00 = pd.read_csv(f'{arguments.data_url}/basic_benchmark_dataset/out1_sample_dimer_new.tsv', sep='\t')
    # pdb_ids0 = list(df00.pdb_id + '_dimer_' + df00.contact_pair)
    
    # restrs = glob.glob(f'{restr_dir}/*/*_raw.pkl')
    pdb_ids = [os.path.basename(i).replace('.pkl', '') for i in os.listdir(raw_feat_dir)]

    # pdb_ids = ['7TIK_dimer_B_C_1v1_5']
    pdb_ids.sort()
    print(len(pdb_ids), pdb_ids[:10])

    # if arguments.ckpt_ids is not None:
    #     ckpt_ids = [int(i)*1000 for i in arguments.ckpt_ids.split('-')]
    # else:
    #     ckpt_ids = [i*1000 for i in [8, 14, 20, 22]] + [22946222]
    ckpt_ids = [0]
        # ckpt_ids = [21946222,]
    ckpt_ids.sort()
    print('ckpt_ids', ckpt_ids)

    infer_batch(model_gen, data_gen, sn, pdb_ids, ckpt_ids, res_dir, baseline=False, iter=1, num_seed=5, check_tsv_exist=True)
    
    print(f'finish! {datetime.datetime.now()}')
    sn.complete()

    for i in range(3600):
        if sn.check_all_complete():
            break
        time.sleep(60)
        i += 1
        print(f'sleep {i} minute(s)')
        
        
        
