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
# from Bio.SeqUtils import seq1


def get_xl_restraints(seq_dict, restr_file, raw_feature, fdr):

    xlinker_map = {
        'DSBU': 30,
        'CDI': 16,
        'DSSO': 24
    }

    def get_site(cid, pos):
        # print(x)
        # restype, pos, cid, atom = x.split('-', 3)
        pos = int(pos)
        true_res = seq_dict[cid][pos-1]
        # assert seq1(restype) == true_res, (x, true_res)
        abs_pos = sum([len(i) for i in list(seq_dict.values())[:list(seq_dict.keys()).index(cid)]])+pos-1
        aatype = int(raw_feature['aatype'][abs_pos])
        assert restypes[aatype] == true_res, [aatype, restypes[aatype], true_res]
        return abs_pos, true_res

    def get_distri(cutoff, fdr):
        xbool = np.concatenate([BINS, [np.inf]])<=cutoff
        x = np.ones(len(BINS)+1)
        x[xbool] = (1-fdr) * (x[xbool]/x[xbool].sum())
        x[~xbool] = fdr * (x[~xbool]/x[~xbool].sum())
        assert x[xbool].max() > x[~xbool].max(), (x[xbool].max(), x[~xbool].max())
        return x

    seqlen = int(np.sum([len(s) for s in seq_dict.values()]))
    sbr = np.zeros((seqlen, seqlen, len(BINS) + 1))
    sbr_mask = np.zeros((seqlen, seqlen))
    interface_mask = np.zeros(seqlen)
    
    with open(restr_file, 'rb') as f:
        restr = pickle.load(f)

    for xlinker, xls in restr.items():
        for xl in xls:
            cid1, pos1, cid2, pos2 = xl.split('-')
            posi, resi = get_site(cid1, pos1)
            posj, resj = get_site(cid2, pos2)
            print(f'xlinker: {xlinker}, {cid1}-{pos1}-{resi} <====> {cid2}-{pos2}-{resj}')
            distri = get_distri(cutoff=xlinker_map[xlinker], fdr=fdr)
            sbr[posi, posj] = distri
            sbr[posj, posi] = distri
            sbr_mask[posi, posj] = 1
            sbr_mask[posj, posi] = 1
    restraints = {'sbr': sbr, 'sbr_mask': sbr_mask, 'interface_mask': interface_mask}
    return restraints

def get_cdr(seq_dict, cdr_file):
    with open(cdr_file, 'rb') as f:
        cdr = pickle.load(f)
    seqlens = [len(s) for s in seq_dict.values()]
    seqlen = sum(seqlens)
    interface_mask = np.zeros(seqlen)
    for cid, cdrs in cdr.items():
        before_len = sum(seqlens[:list(seq_dict.keys()).index(cid)])
        for cdr_idx in cdrs:
            pos = before_len + cdr_idx
            interface_mask[pos] = 1
    return interface_mask


class MyDataGenerator(DataGenerator):
    def __init__(self, raw_feat_dir, fasta_dir, restr_dir, cdr_dir, reorder=False):
        super().__init__(raw_feat_dir, fasta_dir, reorder)
        self.restr_dir = restr_dir
        self.cdr_dir = cdr_dir
        self.use_cdr = False

    def get_pattern(self, pdb_id):
        pdb_id = pdb_id.split('_fdr')[0]
        pat_dict = super().get_pattern(pdb_id)
        restr_pat = f'{self.restr_dir}/{pdb_id}.pkl'
        cdr_pat = f'{self.cdr_dir}/{pdb_id}_cdr.pkl'
        pat_dict.update({'restr': restr_pat, 'cdr': cdr_pat})
        return pat_dict
    
    def get_data(self, pdb_id):
        fdr = float(pdb_id.split('_fdr')[-1])
        raw_feature = self.get_feat(pdb_id)
        seq_dict = self.get_seqs_dict(pdb_id)
        seq_dict = {k.split('_')[-1]:v for k, v in seq_dict.items()}
        restr_file = self.get_files(pdb_id)['restr']
        restraints =  get_xl_restraints(seq_dict, restr_file, raw_feature, fdr)
        if self.use_cdr:
            restraints['interface_mask'] = get_cdr(seq_dict, self.get_files(pdb_id)['cdr'])
        restraints['asym_id'] = raw_feature['asym_id']
        os.makedirs(f'/job/file/abxl_restr', exist_ok=True)
        if not os.path.exists(f'/job/file/abxl_restr/{pdb_id}_restr.pkl'):
            with open(f'/job/file/abxl_restr/{pdb_id}_restr.pkl', 'wb') as f:
                pickle.dump(restraints, f)
        return raw_feature, restraints
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Inputs for eval.py')
    parser.add_argument('--train_url', required=False, default="/job/output/", help='Location of training output')
    parser.add_argument('--data_url', required=False, default="/job/dataset/", help='Location of data')
    parser.add_argument('--data_config', default="/job/file/config/data-infer.yaml", help='data process config')
    parser.add_argument('--model_config', default="/job/file/config/model-infer.yaml", help='model config')
    parser.add_argument('--ckpt_dir', default="ft-grasp-v9-lv-64", help='model config')
    parser.add_argument('--seq_len', default=1280, type=int)
    parser.add_argument('--mixed_precision', default=1, type=int)
    parser.add_argument('--multimer', default=1, type=int)
    # parser.add_argument('--uniform', default=1, type=int)
    # parser.add_argument('--nbdist', default=10.0, type=float)
    parser.add_argument('--jobname', default='abxl', type=str)
    # parser.add_argument('--fdr', default=0.05, type=float)
    parser.add_argument('--ckpt_ids', default=None, type=str)
    
    
    arguments = parser.parse_args()
    print(arguments)
    
    # set path
    raw_feat_dir = f'{arguments.data_url}/xl_antibody/raw_feat'
    fasta_dir = f'{arguments.data_url}/xl_antibody/fasta'
    restr_dir = f'{arguments.data_url}/xl_antibody/xl'
    cdr_dir = f'{arguments.data_url}/xl_antibody/cdr'
    res_dir = f'{arguments.train_url}/grasp_infer_res/{arguments.ckpt_dir}_{arguments.jobname}'
    ckpt_dir = f'{arguments.train_url}/ckpt_dir/{arguments.ckpt_dir}'
    
    data_gen = MyDataGenerator(raw_feat_dir, fasta_dir, restr_dir, cdr_dir)
    model_gen = ModelGenerator(arguments, ckpt_dir)
    sn = infer_config(rotate_split=False, outdir=res_dir)
    sn.start_job()
    
    
    pdb_ids = [i.split('.fasta')[0] for i in os.listdir(fasta_dir)]
    pdb_ids = [i for i in pdb_ids if i != '6OGE']
    pdb_ids = [f'{i}_fdr{j}' for i in pdb_ids for j in [0.05,]]
    print('pdb_ids', pdb_ids)

    if arguments.ckpt_ids is not None:
        ckpt_ids = [int(i)*1000 for i in arguments.ckpt_ids.split('-')]
    else:
        ckpt_ids = [i*1000 for i in [8, 14, 20, 22]] + [22946222]
    print('ckpts', ckpt_ids)
    infer_batch(model_gen, data_gen, sn, pdb_ids, ckpt_ids, res_dir)
    
    print(f'finish! {datetime.datetime.now()}')
    sn.complete()
    
    for i in range(3600):
        if sn.check_all_complete():
            break
        time.sleep(60)
        i += 1
        print(f'sleep {i} minute(s)')
        
        
        
