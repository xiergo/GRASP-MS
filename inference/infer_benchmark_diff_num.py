import argparse
import os
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
    def __init__(self, raw_feat_dir, restr_dir):
        super().__init__(raw_feat_dir)
        self.restr_dir = restr_dir

    def get_pattern(self, pdb_id):
        pdb_id0, inter_num, intra_num, interface_num = pdb_id.split('_')
        pat_dict = super().get_pattern(pdb_id0)
        restr_pat = f'{self.restr_dir}/{pdb_id0}_contact.pkl'
        pat_dict.update({'restr': restr_pat})
        return pat_dict
    
    def get_data(self, pdb_id):
        _, inter_num, intra_num, interface_num = pdb_id.split('_')
        inter_num = int(inter_num)
        intra_num = int(intra_num)
        interface_num = int(interface_num)
        raw_feature = self.get_feat(pdb_id)  
        restr_file = self.get_files(pdb_id)['restr']
        with open(restr_file, 'rb') as f:
            contact_feat = pickle.load(f)
        if (inter_num != 0) and (interface_num != 0):
            restraints = generate_restraints(contact_feat, inter_num, 0, 0)
            restraints['interface_mask'] = generate_restraints(contact_feat, 0, 0, interface_num)['interface_mask']
        else:
            restraints = generate_restraints(contact_feat, inter_num, intra_num, interface_num)
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
    parser.add_argument('--jobname', default='benchmark_diff_num', type=str)
    parser.add_argument('--ckpt_ids', default=None, type=str)
    
    arguments = parser.parse_args()
    print(arguments)
    
    # set path
    raw_feat_dir = f'{arguments.data_url}/raw_feat'
    restr_dir = f'{arguments.data_url}/contact_info/'
    res_dir = f'{arguments.train_url}/grasp_infer_res/{arguments.ckpt_dir}_{arguments.jobname}'
    ckpt_dir = f'{arguments.train_url}/ckpt_dir/{arguments.ckpt_dir}'
    
    data_gen = MyDataGenerator(raw_feat_dir, restr_dir)
    model_gen = ModelGenerator(arguments, ckpt_dir)
    sn = infer_config(rotate_split=False, outdir=res_dir)
    sn.start_job()

    with open(f'{arguments.data_url}/benchmark_dataset_hardtarget.tsv', 'r') as f:
        all_pdbs = [x.split('\t')[0] for i, x in enumerate(f.readlines()) if (i >= 1)]
    assert len(all_pdbs) == 48

    sets = [2, 5, 10, 20]
    # pdb_ids = [f'{i}_{j}_0_0' for i in all_pdbs for j in sets] + [f'{i}_0_0_{j}' for i in all_pdbs for j in sets] + [f'{i}_5_0_5' for i in all_pdbs]
    pdb_ids = [f'{i}_5_0_10' for i in all_pdbs]
    
    if arguments.ckpt_ids is not None:
        ckpt_ids = [int(i)*1000 for i in arguments.ckpt_ids.split('-')]
    else:
        ckpt_ids = [i*1000 for i in [8, 14, 20, 22]]
    ckpt_ids.sort()
    print('ckpt_ids', ckpt_ids)

    infer_batch(model_gen, data_gen, sn, pdb_ids, ckpt_ids, res_dir, baseline=False, iter=5, num_seed=1, check_tsv_exist=True)
    
    print(f'finish! {datetime.datetime.now()}')
    sn.complete()

    for i in range(3600):
        if sn.check_all_complete():
            break
        time.sleep(60)
        i += 1
        print(f'sleep {i} minute(s)')
        
        
        
