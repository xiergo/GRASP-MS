import argparse
import os
import glob
import time
import datetime
import pandas as pd
import numpy as np
import pickle
from restraint_sample import BINS
from utils_infer import infer_config, infer_batch, DataGenerator, ModelGenerator



def get_restraints(restr_file):
    with open(restr_file, 'rb') as f:
        restraints = pickle.load(f)
    return restraints

# def parse_restr(restr_file):
#     ls = []
#     with open(restr_file, 'r') as f:
#         lines = f.readlines()
#     for line in lines:
#         a, cid = line.strip().split('-')
#         ls.append({
#             'res':  a[0],
#             'pos': int(a[1:]),
#             'cid': cid
#         })
#     return pd.DataFrame(ls)
 
# def get_interface(restr_file, mapping):
#     print('mapping:', mapping)
#     df = parse_restr(restr_file)
#     print(df)
#     print(f'total sites: {df.shape[0]}')
#     seqlens = [len(i) for i in mapping.values()]
#     seqlen = sum(seqlens)
#     bins = BINS
#     sbr = np.zeros((seqlen, seqlen, len(bins) + 1))
#     sbr_mask = np.zeros((seqlen, seqlen))
#     interface_mask = np.zeros(seqlen)
    
#     for _, r in df.iterrows():
#         assert r.res == mapping[r.cid][r.pos-1], (r.res, mapping[r.cid][r.pos-1])
#         start_pos = sum(seqlens[:list(mapping.keys()).index(r.cid)])
#         interface_mask[r.pos+start_pos-1] = 1
#     restraints =  {'sbr': sbr, 'sbr_mask': sbr_mask, 'interface_mask': interface_mask}
#     return restraints


class MyDataGenerator(DataGenerator):
    def __init__(self, raw_feat_dir, fasta_dir, restr_dir, reorder=False):
        super().__init__(raw_feat_dir, fasta_dir, reorder)
        self.restr_dir = restr_dir

    def get_pattern(self, pdb_id):
        pdb_id0, _ = pdb_id.split('_usesbr')
        pat_dict = super().get_pattern(pdb_id0)
        restr_pat = f'{self.restr_dir}/{pdb_id}_restraints.pkl'
        pat_dict.update({'restr': restr_pat})
        return pat_dict
        
    def get_data(self, pdb_id):
        raw_feature = self.get_feat(pdb_id)
        restr_file = self.get_files(pdb_id)['restr']
        restraints = get_restraints(restr_file)
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
    parser.add_argument('--jobname', default='csp_ab', type=str)
    parser.add_argument('--ckpt_ids', default=None, type=str)
    parser.add_argument('--baseline', default=0, type=int)
    parser.add_argument('--quick', default=0, type=int)
    
    arguments = parser.parse_args()
    print(arguments)
    
    # set path
    raw_feat_dir = f'{arguments.data_url}/csp_ab/raw_feat'
    fasta_dir = f'{arguments.data_url}/csp_ab/fasta'
    restr_dir = f'{arguments.data_url}/csp_ab/restraint'
    res_dir = f'{arguments.train_url}/grasp_infer_res/{arguments.ckpt_dir}_{arguments.jobname}'
    ckpt_dir = f'{arguments.train_url}/ckpt_dir/{arguments.ckpt_dir}'
    
    data_gen = MyDataGenerator(raw_feat_dir, fasta_dir, restr_dir)
    model_gen = ModelGenerator(arguments, ckpt_dir)
    sn = infer_config(rotate_split=False, outdir=res_dir)
    sn.start_job()
    
    pdb_ids = ['4G6J', '4G6M']
    pdb_ids = [f'{pdb_id}_usesbr{j}' for pdb_id in pdb_ids for j in [0, 1, 2]]

    if arguments.ckpt_ids is not None:
        ckpt_ids = [int(i)*1000 for i in arguments.ckpt_ids.split('-')]
    else:
        ckpt_ids = [i*1000 for i in (8, 14, 20, 22)] + [22946222]
    ckpt_ids.sort()
    quick = bool(arguments.quick)
    infer_batch(model_gen, data_gen, sn, pdb_ids, ckpt_ids, res_dir, baseline=bool(arguments.baseline), num_seed=5, check_tsv_exist=True, quick=quick)
    
    print(f'finish! {datetime.datetime.now()}')
    sn.complete()

    for i in range(3600):
        if sn.check_all_complete():
            break
        time.sleep(60)
        i += 1
        print(f'sleep {i} minute(s)')
        
        
        
