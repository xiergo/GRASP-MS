import argparse
import os
import re
import pickle
import glob
import time
import datetime
import numpy as np
import pandas as pd
from restraint_sample import BINS
from utils_infer import infer_config, infer_batch, DataGenerator, ModelGenerator
from mindsponge1.common.residue_constants import restypes
from Bio.SeqUtils import seq1

def get_seq_dict(fasta):
    with open(fasta, 'r') as f:
        cont = f.readlines()
    seqs = [i.strip() for i in cont[1::2]]
    cids = [i.strip().split('_')[-1][0] for i in cont[::2]]
    seq_dict = {i:j for i,j in zip(cids, seqs)}
    return seq_dict

def get_xl_restraints(seq_dict, restr_file, raw_feature, fdr, cheat_mode):

    def get_site(x):
        # print(x)
        restype, pos, cid, atom = x.split('-', 3)
        pos = int(pos)
        true_res = seq_dict[cid][pos-1]
        assert seq1(restype) == true_res, (x, true_res)
        abs_pos = sum([len(i) for i in list(seq_dict.values())[:list(seq_dict.keys()).index(cid)]])+pos-1
        aatype = int(raw_feature['aatype'][abs_pos])
        assert restypes[aatype] == true_res, [aatype, restypes[aatype], true_res]
        return abs_pos

    def get_distri(cutoff, fdr):
        xbool = np.concatenate([BINS, [np.inf]])<=cutoff
        x = np.ones(len(BINS)+1)
        x[xbool] = (1-fdr) * (x[xbool]/x[xbool].sum())
        x[~xbool] = fdr * (x[~xbool]/x[~xbool].sum())
        assert x[xbool].max() > x[~xbool].max(), (x[xbool].max(), x[~xbool].max())
        return x

    seqlen = int(np.sum([len(s) for s in seq_dict.values()]))

    df = pd.read_csv(restr_file, sep='\t')

    sbr = np.zeros((seqlen, seqlen, len(BINS) + 1))
    sbr_mask = np.zeros((seqlen, seqlen))
    interface_mask = np.zeros(seqlen)
    
    for _, row in df.iterrows():
        if cheat_mode and (float(row.euc_dist) > 25):
            continue
        posi, posj = [get_site(i) for i in [row.aa1, row.aa2]]
        distri = get_distri(25, fdr)
        sbr[posi, posj] = distri
        sbr[posj, posi] = distri
        sbr_mask[posi, posj] = 1
        sbr_mask[posj, posi] = 1
    restraints = {'sbr': sbr, 'sbr_mask': sbr_mask, 'interface_mask': interface_mask}
    return restraints
 

class MyDataGenerator(DataGenerator):
    def __init__(self, raw_feat_dir, fasta_dir, restr_dir, reorder=False):
        super().__init__(raw_feat_dir, fasta_dir, reorder)
        self.restr_dir = restr_dir

    def get_pattern(self, pdb_id):
        pdb_id, rep_id, _, _ = pdb_id.split('_')
        pat_dict = super().get_pattern(pdb_id)
        restr_pat = f'{self.restr_dir}/{pdb_id}_rep{rep_id}.tsv'
        pat_dict.update({'restr': restr_pat})
        return pat_dict
    
    def get_data(self, pdb_id):
        raw_feature = self.get_feat(pdb_id)
        _, _, fdr, cheat = pdb_id.split('_')
        fdr = float(fdr.replace('fdr', ''))
        cheat = int(cheat.replace('cheat', ''))
        seq_dict = self.get_seqs_dict(pdb_id)
        restr_file = self.get_files(pdb_id)['restr']
        restraints =  get_xl_restraints(seq_dict, restr_file, raw_feature, fdr=fdr, cheat_mode=cheat)
        restraints['asym_id'] = raw_feature['asym_id']
        return raw_feature, restraints
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Inputs for eval.py')
    parser.add_argument('--train_url', required=False, default="/job/output/", help='Location of training output')
    parser.add_argument('--data_url', required=False, default="/job/dataset/", help='Location of data')
    parser.add_argument('--data_config', default="/job/file/config/data-infer.yaml", help='data process config')
    parser.add_argument('--model_config', default="/job/file/config/model-infer.yaml", help='model config')
    parser.add_argument('--ckpt_dir', default="ft-grasp-v9-lv-64", help='model config')
    parser.add_argument('--seq_len', default=1024, type=int)
    parser.add_argument('--mixed_precision', default=1, type=int)
    parser.add_argument('--multimer', default=1, type=int)
    parser.add_argument('--jobname', default='simxl', type=str)
    # parser.add_argument('--cheat_mode', default=0, type=int)
    # parser.add_argument('--fdr', default=0.05, type=float)
    parser.add_argument('--ckpt_ids', default=None, type=str)
    
    
    arguments = parser.parse_args()
    print(arguments)
    
    # set path
    raw_feat_dir = f'{arguments.data_url}/raw_feat'
    fasta_dir = f'{arguments.data_url}/xl_simulated/fasta'
    restr_dir = f'{arguments.data_url}/xl_simulated/benchmark_xl_hard/'
    res_dir = f'{arguments.train_url}/grasp_infer_res/{arguments.ckpt_dir}_{arguments.jobname}'
    ckpt_dir = f'{arguments.train_url}/ckpt_dir/{arguments.ckpt_dir}'
    
    data_gen = MyDataGenerator(raw_feat_dir, fasta_dir, restr_dir)
    model_gen = ModelGenerator(arguments, ckpt_dir)
    sn = infer_config(rotate_split=False, outdir=res_dir)
    sn.start_job()
    
    with open(f'{restr_dir}/../xl_hard_1024.list', 'r') as f:
        pdb_ids = [i.split()[0] for i in f.readlines()]
    pdb_ids = [f'{pdb_id}_{rep}_fdr{fdr}_cheat{cheat}' for pdb_id in pdb_ids \
                for rep in range(3) \
                for fdr in [0.05]\
                for cheat in [0, ]]
    if arguments.ckpt_ids is not None:
        ckpt_ids = [int(i)*1000 for i in arguments.ckpt_ids.split('-')]
    else:
        ckpt_ids = [i*1000 for i in range(8, 27, 2)]
    ckpt_ids.sort()
    print('ckpt_ids', ckpt_ids)
    infer_batch(model_gen, data_gen, sn, pdb_ids, ckpt_ids, res_dir)
    
    print(f'finish! {datetime.datetime.now()}')
    sn.complete()
    
    for i in range(3600):
        if sn.check_all_complete():
            break
        time.sleep(60)
        i += 1
        print(f'sleep {i} minute(s)')
        
        
        
