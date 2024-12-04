import argparse
import os
import glob
import time
import pickle
import datetime
import pandas as pd
import numpy as np
from restraint_sample import BINS
from utils_infer import infer_config, infer_batch, DataGenerator, ModelGenerator

def get_colabdock_mvn_interface(pkl_path, seqlen):
    with open(pkl_path, 'rb') as f:
        pos = pickle.load(f)
    pos = list(pos[0][0][1])+list(pos[1][0][1])
    pos.sort()
    pos = [i-1 for i in pos]
    bins = BINS
    sbr = np.zeros((seqlen, seqlen, len(bins) + 1))
    sbr_mask = np.zeros((seqlen, seqlen))
    print(f'{len(pos)} interfaces: {pos}')
    interface_mask = np.zeros(seqlen)
    interface_mask[pos] = 1
    restraints =  {'sbr': sbr, 'sbr_mask': sbr_mask, 'interface_mask': interface_mask}
    return restraints


def generate_split_first_num(split_file):
    with open(split_file, 'r') as f:
        split = [i.strip().split(',') for i in f.readlines()]
    print(split)
    return len(split[0])

def generate_index(lenlsls):
    ls = []
    for lenls in lenlsls:
        last = 0
        for l in lenls:
            a = np.arange(l)+last
            ls.append(a)
            last = a[-1]+200
    return np.concatenate(ls, axis=0)

def generate_asym_id(lenlsls):
    return np.repeat(np.arange(len(lenlsls))+1, [np.sum(i) for i in lenlsls])


def dict_update_keepdtype(d1, d2):
    for k, v in d2.items():
        if k in d1:
            d1[k] = v.astype(d1[k].dtype)
        
def update_feature_make_two_chains(feat, first_num):
    lenls = np.unique(feat['asym_id'],return_counts=True)[1]
    lenlsls = [lenls[:first_num], lenls[first_num:]]
    asym_id = generate_asym_id(lenlsls)
    d_update = {
        'residue_index': generate_index(lenlsls),
        'asym_id': asym_id,
        'entity_id': asym_id,
        'assembly_num_chains': np.array(2)
    }
    dict_update_keepdtype(feat, d_update)


class MyDataGenerator(DataGenerator):
    def __init__(self, raw_feat_dir, fasta_dir, restr_dir, reorder=False):
        super().__init__(raw_feat_dir, fasta_dir, reorder)
        self.restr_dir = restr_dir

    def get_pattern(self, pdb_id):
        pdb_id, rep_id, _ = pdb_id.split('_')
        pat_dict = super().get_pattern(pdb_id)
        restr_pat = f'{self.restr_dir}/{pdb_id}/restraints/restraints_surface_pickleloaded_{rep_id}.pkl'
        split_file = f'{self.raw_feat_dir}/../split/{pdb_id}.txt'
        pat_dict.update({'restr': restr_pat, 'split': split_file})
        return pat_dict
    
    def get_data(self, pdb_id):
        dimer = int(pdb_id.split('_')[-1].replace('dimer', ''))
        raw_feature = self.get_feat(pdb_id)
        if dimer:
            split_file = self.get_files(pdb_id)['split']
            first_num = generate_split_first_num(split_file)
            update_feature_make_two_chains(raw_feature, first_num)
        
        ori_res_length = raw_feature['msa'].shape[1]

        restr_file = self.get_files(pdb_id)['restr']
        restraints = get_colabdock_mvn_interface(restr_file, ori_res_length)
        restraints['asym_id'] = raw_feature['asym_id']
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
    parser.add_argument('--jobname', default='colabdock_ab_cleaned', type=str)
    # parser.add_argument('--dimer_mode', default=0, type=int)
    parser.add_argument('--ckpt_ids', default=None, type=str)
    
    arguments = parser.parse_args()
    print(arguments)
    
    # set path
    raw_feat_dir = f'{arguments.data_url}/colabdock/features'
    fasta_dir = f'{arguments.data_url}/colabdock/fasta'
    restr_dir = f'{arguments.data_url}/colabdock/Antibody_Release'
    res_dir = f'{arguments.train_url}/grasp_infer_res/{arguments.ckpt_dir}_{arguments.jobname}'
    ckpt_dir = f'{arguments.train_url}/ckpt_dir/{arguments.ckpt_dir}'
    
    data_gen = MyDataGenerator(raw_feat_dir, fasta_dir, restr_dir, reorder=False)
    model_gen = ModelGenerator(arguments, ckpt_dir)
    sn = infer_config(rotate_split=False, outdir=res_dir)
    sn.start_job()
    
    pdb_ids = [f'{i}_{j}_dimer{dimer}' \
               for i in os.listdir(restr_dir) if len(i)==4 \
               for j in [1, 2, 3]\
                for dimer in [0, 1]]

    if arguments.ckpt_ids is not None:
        ckpt_ids = [int(i)*1000 for i in arguments.ckpt_ids.split('-')]
    else:
        ckpt_ids = [i*1000 for i in [8, 14, 20, 22]] + [22946222]
    ckpt_ids.sort()
    print('ckpt_ids', ckpt_ids)
    infer_batch(model_gen, data_gen, sn, pdb_ids, ckpt_ids, res_dir, num_seed=5)
    
    print(f'finish! {datetime.datetime.now()}')
    sn.complete()

    for i in range(3600):
        if sn.check_all_complete():
            break
        time.sleep(60)
        i += 1
        print(f'sleep {i} minute(s)')
        
        
        
