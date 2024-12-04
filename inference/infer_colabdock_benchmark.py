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

def get_colabdock_1v1_interface(pkl_path, seqlen):
    with open(pkl_path, 'rb') as f:
        pos = pickle.load(f)
    if not isinstance(pos[0], list):
        pos = [pos]
    
    bins = BINS
    sbr = np.zeros((seqlen, seqlen, len(bins) + 1))
    sbr_mask = np.zeros((seqlen, seqlen))
    interface_mask = np.zeros(seqlen)
    
    print(len(pos), 'residue pairs ---')
    for i, j in pos:
        contact = np.zeros(len(bins)+1)
        contact[0] = 1.0
        sbr[i-1, j-1] = contact
        sbr[j-1, i-1] = contact
        sbr_mask[i-1, j-1] = 1.0
        sbr_mask[j-1, i-1] = 1.0
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
    
    def get_data(self, pdb_id):
        pdb_id, rep_id = pdb_id.split('_')
        raw_feature = self.get_feat(pdb_id)
        if arguments.dimer_mode:
            split_file = f'{self.restr_dir}/{pdb_id}/chain_info.txt'
            first_num = generate_split_first_num(split_file)
            update_feature_make_two_chains(raw_feature, first_num)
        
        ori_res_length = raw_feature['msa'].shape[1]

        restr_file = glob.glob(f'{self.restr_dir}/{pdb_id}/restraints/restraints_{rep_id}pairs.pkl')[0]
        restraints = get_colabdock_1v1_interface(restr_file, ori_res_length)
        restraints['asym_id'] = raw_feature['asym_id']
        return raw_feature, restraints
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Inputs for eval.py')
    parser.add_argument('--train_url', required=False, default="/job/output/", help='Location of training output')
    parser.add_argument('--data_url', required=False, default="/job/dataset/", help='Location of data')
    parser.add_argument('--data_config', default="/job/file/config/data-infer.yaml", help='data process config')
    parser.add_argument('--model_config', default="/job/file/config/model-infer.yaml", help='model config')
    parser.add_argument('--ckpt_dir', default="ft-grasp-v6-notfix-nohard-32", help='model config')
    parser.add_argument('--seq_len', default=1024, type=int)
    parser.add_argument('--mixed_precision', default=1, type=int)
    parser.add_argument('--multimer', default=1, type=int)
    parser.add_argument('--jobname', default='colabdock_benchmark1v1', type=str)
    parser.add_argument('--dimer_mode', default=0, type=int)
    
    arguments = parser.parse_args()
    print(arguments)
    
    # set path
    raw_feat_dir = f'{arguments.data_url}/colabdock/release_2024_0115/Benchmark_Release/features'
    fasta_dir = f'{arguments.data_url}/colabdock/release_2024_0115/Benchmark_Release/fasta'
    restr_dir = f'{arguments.data_url}/colabdock/release_2024_0115/Benchmark_Release/1v1'
    res_dir = f'{arguments.train_url}/grasp_infer_res/{arguments.ckpt_dir}_{arguments.jobname}_dimer{arguments.dimer_mode}'
    ckpt_dir = f'{arguments.train_url}/ckpt_dir/{arguments.ckpt_dir}'
    
    data_gen = MyDataGenerator(raw_feat_dir, fasta_dir, restr_dir, reorder=False)
    model_gen = ModelGenerator(arguments, ckpt_dir)
    sn = infer_config(rotate_split=False)
    
    pdb_ids = [f'{i}_{j}' for i in os.listdir(restr_dir) if len(i)==4 for j in [1, 2, 3, 5]]
    ckpt_ids = [i for i in range(0, 200000, 2000) if os.path.isfile(f'{ckpt_dir}/step_{i}.ckpt')]
    # ckpt_ids = [4000,]
    infer_batch(model_gen, data_gen, sn, pdb_ids, ckpt_ids, res_dir, baseline=True, iter=1)
    
    print(f'finish! {datetime.datetime.now()}')
    
    for i in range(120):
        time.sleep(60)
        i += 1
        print(f'sleep {i} minute(s)')
        
        
        
