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
from utils_infer import infer_config, infer_batch, DataGenerator, ModelGenerator, parse_fasta, get_distri
# from Bio.SeqUtils import seq1


def get_xl_restraints(seq_dict, restr_file, raw_feature, fdr):

    seqlen = int(np.sum([len(s) for s in seq_dict.values()]))
    sbr = np.zeros((seqlen, seqlen, len(BINS) + 1))
    sbr_mask = np.zeros((seqlen, seqlen))
    interface_mask = np.zeros(seqlen)
    
    # assert seqlen == raw_feature['aatype'].shape[0], f'{seqlen}!= {raw_feature["aatype"].shape[0]}'

    df = pd.read_csv(restr_file, sep=' ')
    print(df)
    first_chain_len = len(seq_dict[df.columns[0]])
    print(first_chain_len)
    for i, j, d in df.itertuples(index=False, name=None):
        # print(i, j, d)
        distri = get_distri(cutoff=8, fdr=fdr)
        posi = i-1
        posj = first_chain_len+j-1

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
        pdb_id = pdb_id.split('_fdr')[0]
        pat_dict = super().get_pattern(pdb_id)
        restr_pat = f'{self.restr_dir}/{pdb_id}_restr.txt'
        pat_dict.update({'restr': restr_pat})
        return pat_dict
    
    def get_data(self, pdb_id):
        fdr = 0
        raw_feature = self.get_feat(pdb_id)
        seq_dict = parse_fasta(self.get_files(pdb_id)['fasta'])
        restr_file = self.get_files(pdb_id)['restr']
        restraints =  get_xl_restraints(seq_dict, restr_file, raw_feature, fdr)
        restraints['asym_id'] = raw_feature['asym_id']
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
    parser.add_argument('--jobname', default='P10321-P30511', type=str)
    # parser.add_argument('--fdr', default=0.05, type=float)
    parser.add_argument('--ckpt_ids', default=None, type=str)
    
    
    arguments = parser.parse_args()
    print(arguments)
    
    # set path
    raw_feat_dir = f'{arguments.data_url}/P10321-P30511'
    fasta_dir = f'{arguments.data_url}/P10321-P30511'
    restr_dir = f'{arguments.data_url}/P10321-P30511'
    res_dir = f'{arguments.train_url}/grasp_infer_res/{arguments.ckpt_dir}_{arguments.jobname}'
    ckpt_dir = f'{arguments.train_url}/ckpt_dir/{arguments.ckpt_dir}'
    
    data_gen = MyDataGenerator(raw_feat_dir, fasta_dir, restr_dir, reorder=False)
    model_gen = ModelGenerator(arguments, ckpt_dir)
    sn = infer_config(rotate_split=False, outdir=res_dir)
    sn.start_job()
    
    
    pdb_ids = ['P10321-P30511']
    print('pdb_ids', pdb_ids)

    if arguments.ckpt_ids is not None:
        ckpt_ids = [int(i)*1000 for i in arguments.ckpt_ids.split('-')]
    else:
        ckpt_ids = [i*1000 for i in [8, 14, 20, 22]]
    print('ckpts', ckpt_ids)
    infer_batch(model_gen, data_gen, sn, pdb_ids, ckpt_ids, res_dir, viol_thre=4)
    
    print(f'finish! {datetime.datetime.now()}')
    sn.complete()
    
    for i in range(3600):
        if sn.check_all_complete():
            break
        time.sleep(60)
        i += 1
        print(f'sleep {i} minute(s)')
        
        
        
