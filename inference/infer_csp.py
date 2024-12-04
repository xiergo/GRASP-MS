import argparse
import os
import glob
import time
import datetime
import pandas as pd
import numpy as np
from restraint_sample import BINS
from utils_infer import infer_config, infer_batch, DataGenerator, ModelGenerator


def get_distri(cutoff, fdr):
    xbool = np.concatenate([BINS, [np.inf]])<=cutoff
    x = np.ones(len(BINS)+1)
    x[xbool] = (1-fdr) * (x[xbool]/x[xbool].sum())
    x[~xbool] = fdr * (x[~xbool]/x[~xbool].sum())
    assert x[xbool].max() > x[~xbool].max(), (x[xbool].max(), x[~xbool].max())
    return x


def get_site(mapping, r):
    # r: residue name and position, e.g. C10-A, residue name is C, position is 10, chain id is A
    a, cid = r.strip().split('-')
    res = a[0]
    pos = int(a[1:])
    seqlens = [len(i) for i in mapping.values()]
    assert res == mapping[cid][pos-1], (res, mapping[cid][pos-1])
    start_pos = sum(seqlens[:list(mapping.keys()).index(cid)])
    return start_pos + pos - 1


def get_restraints(restr_file, mapping):
    print('mapping:', mapping)
    seqlens = [len(i) for i in mapping.values()]
    seqlen = sum(seqlens)
    sbr = np.zeros((seqlen, seqlen, len(BINS) + 1))
    sbr_mask = np.zeros((seqlen, seqlen))
    interface_mask = np.zeros(seqlen)
    with open(restr_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        restr = line.strip().split(',')
        if len(restr)== 3:
            r1, r2, cutoff = restr
            r1_pos = get_site(mapping, r1)
            r2_pos = get_site(mapping, r2)
            distr = get_distri(int(cutoff), 0)
            sbr[r1_pos, r2_pos] = distr
            sbr[r2_pos, r1_pos] = distr
            sbr_mask[r1_pos, r2_pos] = 1
            sbr_mask[r2_pos, r1_pos] = 1
        elif len(restr)== 1:
            r1 = restr[0]
            r1_pos = get_site(mapping, r1)
            interface_mask[r1_pos] = 1
        else:
            raise ValueError(f'wrong restraint format: {line}')
    
    restraints =  {'sbr': sbr, 'sbr_mask': sbr_mask, 'interface_mask': interface_mask}
    return restraints


class MyDataGenerator(DataGenerator):
    def __init__(self, raw_feat_dir, fasta_dir, restr_dir, reorder=False):
        super().__init__(raw_feat_dir, fasta_dir, reorder)
        self.restr_dir = restr_dir

    def get_pattern(self, pdb_id):
        pdb_id0, version = pdb_id.split('-ver-')
        
        pat_dict = {
            'raw_feat': f'{self.raw_feat_dir}/{pdb_id0}.pkl',
            'fasta': f'{self.fasta_dir}/{pdb_id0}.fasta',
            'restr': f'{self.restr_dir}/{pdb_id}.txt'
        }
        return pat_dict
        
    def get_data(self, pdb_id):
        # _, use_sbr, _ = pdb_id.split('-')
        raw_feature = self.get_feat(pdb_id)
        mapping = self.get_seqs_dict(pdb_id)
        mapping = {k[-1]: v for k, v in mapping.items()}
        restr_file = self.get_files(pdb_id)['restr']
        restraints = get_restraints(restr_file, mapping)
        restraints['asym_id'] = raw_feature['asym_id']
        # if int(use_sbr)>0:
        #     print('generate sbr restraints from interface')
        #     interchain_mask = (restraints['asym_id'][:, None] != restraints['asym_id'][None])
        #     cutoff = 10
        #     x = (np.concatenate((BINS, [10000]))<=cutoff).astype(np.float32)
        #     x /= x.sum()
        #     print(x)
        #     restraints['sbr_mask'] = (restraints['interface_mask'][:, None] * restraints['interface_mask'][None]) * interchain_mask
        #     restraints['sbr'][restraints['sbr_mask']>0.5] = x
        #     print(restraints['sbr_mask'].sum()/2)
        #     if int(use_sbr)==1:
        #         print('not use interface restraints')
        #         restraints['interface_mask'] *= 0
        return raw_feature, restraints
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Inputs for eval.py')
    parser.add_argument('--train_url', required=False, default="/job/output/", help='Location of training output')
    parser.add_argument('--data_url', required=False, default="/job/dataset/", help='Location of data')
    parser.add_argument('--data_config', default="/job/file/config/data-infer.yaml", help='data process config')
    parser.add_argument('--model_config', default="/job/file/config/model-infer.yaml", help='model config')
    parser.add_argument('--ckpt_dir', default="ft-grasp-v11-64", help='model config')
    parser.add_argument('--seq_len', default=256, type=int)
    parser.add_argument('--mixed_precision', default=1, type=int)
    parser.add_argument('--multimer', default=1, type=int)
    parser.add_argument('--jobname', default='csp_yq', type=str)
    parser.add_argument('--ckpt_ids', default=None, type=str)
    parser.add_argument('--baseline', default=0, type=int)
    
    arguments = parser.parse_args()
    print(arguments)
    
    # set path
    raw_feat_dir = f'{arguments.data_url}/csp/raw_feat'
    fasta_dir = f'{arguments.data_url}/csp/fasta'
    restr_dir = f'{arguments.data_url}/csp/restr'
    res_dir = f'{arguments.train_url}/grasp_infer_res/{arguments.ckpt_dir}_{arguments.jobname}'
    ckpt_dir = f'{arguments.train_url}/ckpt_dir/{arguments.ckpt_dir}'
    
    data_gen = MyDataGenerator(raw_feat_dir, fasta_dir, restr_dir)
    model_gen = ModelGenerator(arguments, ckpt_dir)
    sn = infer_config(rotate_split=False, outdir=res_dir)
    sn.start_job()
    
    # pdb_ids = [f'{pdb_id}-{i}-{j}' for i in [0, 1, 2] for j in [2, 3] for pdb_id in ['psi']]
    pdb_ids = ['psi_trim-ver-7']
    if arguments.ckpt_ids is not None:
        ckpt_ids = [int(i)*1000 for i in arguments.ckpt_ids.split('-')]
    else:
        ckpt_ids = [i*1000 for i in (8, 14, 20, 22)] + [22946222]
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
        
        
        
