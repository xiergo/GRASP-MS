import argparse
import os
import re
import pickle
import glob
import time
import datetime
import numpy as np
from restraint_sample import BINS
from utils_infer import infer_config, infer_batch, DataGenerator, ModelGenerator
from mindsponge1.common.residue_constants import restypes
from Bio.SeqUtils import seq1

def get_seq_dict(fasta):
    with open(fasta, 'r') as f:
        cont = f.readlines()
    seqs = [i.strip() for i in cont[1::2]]
    cids = [re.search('>\w+_(.*)\|', i).group(1) for i in cont[::2]]
    seq_dict = {i:j for i,j in zip(cids, seqs)}
    return seq_dict

def get_xl_restraints(pdb_id, seq_dict, restr_file, raw_feature, uniform):

    def get_site(x):
        restype, pos, cid, atom = x.split('-')
        pos = int(pos) + restr_dict['sequence_index'][cid]
        if cid not in seq_dict:
            return None
        true_res = seq_dict[cid][pos-1]
        assert seq1(restype) == true_res, (x, true_res)
        abs_pos = sum([len(i) for i in list(seq_dict.values())[:list(seq_dict.keys()).index(cid)]])+pos-1
        aatype = int(raw_feature['aatype'][abs_pos])
        assert restypes[aatype] == true_res, [aatype, restypes[aatype], true_res]
        return abs_pos

    def get_distri(linker, uniform):
        if uniform:
            cutoff_dict = {'DHSO': 30, 'DSSO':30, 'DMTMM': 25}
            cutoff = cutoff_dict[linker]
            x = (np.concatenate([BINS, [np.inf]])<=cutoff)
        else:
            # TODO
            raise NotImplementedError
        x = x/x.sum()
        return x

    seqlen = int(np.sum([len(s) for s in seq_dict.values()]))

    with open(restr_file, 'rb') as f:
        restr_dict = pickle.load(f)[f'{pdb_id}.pdb']

    sbr = np.zeros((seqlen, seqlen, len(BINS) + 1))
    sbr_mask = np.zeros((seqlen, seqlen))
    interface_mask = np.zeros(seqlen)
    for key in ['inter_list', 'intra_list']:
        x = restr_dict[key]
        for s, l, d in zip(*x.values()):
            posi, posj = [get_site(i) for i in s.split(':')]
            if (posi is None) or (posj is None):
                continue
            distri = get_distri(l, uniform)
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
    
    def get_data(self, pdb_id):
        raw_feature = self.get_feat(pdb_id)
        fasta_file = glob.glob(f'{self.fasta_dir}/{pdb_id}*fasta')[0]
        seq_dict = get_seq_dict(fasta_file)
        #    restr_file: f'{arguments.data_url}/dms_data_full_antigen/dms/{pdb_id}_dms.tsv'    
        restr_file = f'{self.restr_dir}/meta_data_select.pkl'
        restraints =  get_xl_restraints(pdb_id, seq_dict, restr_file, raw_feature, arguments.uniform)
        restraints['asym_id'] = raw_feature['asym_id']
        return raw_feature, restraints
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Inputs for eval.py')
    parser.add_argument('--train_url', required=False, default="/job/output/", help='Location of training output')
    parser.add_argument('--data_url', required=False, default="/job/dataset/", help='Location of data')
    parser.add_argument('--data_config', default="/job/file/config/data-infer.yaml", help='data process config')
    parser.add_argument('--model_config', default="/job/file/config/model-infer.yaml", help='model config')
    parser.add_argument('--ckpt_dir', default="ft-grasp-v6-notfix-nohard-32", help='model config')
    parser.add_argument('--seq_len', default=1280, type=int)
    parser.add_argument('--mixed_precision', default=1, type=int)
    parser.add_argument('--multimer', default=1, type=int)
    parser.add_argument('--uniform', default=1, type=int)
    parser.add_argument('--jobname', default='xl_high', type=str)
    
    
    arguments = parser.parse_args()
    print(arguments)
    
    # set path
    raw_feat_dir = f'{arguments.data_url}/xl_high/features'
    fasta_dir = f'{arguments.data_url}/xl_high/fasta'
    restr_dir = f'{arguments.data_url}/xl_high/'
    res_dir = f'{arguments.train_url}/grasp_infer_res/{arguments.ckpt_dir}_{arguments.jobname}_{arguments.uniform}'
    ckpt_dir = f'{arguments.train_url}/ckpt_dir/{arguments.ckpt_dir}'
    
    data_gen = MyDataGenerator(raw_feat_dir, fasta_dir, restr_dir)
    model_gen = ModelGenerator(arguments, ckpt_dir)
    sn = infer_config(rotate_split=False)
    
    pdb_ids = [i.split('.')[0] for i in os.listdir(fasta_dir)]
    pdb_ids = ['1jeq', '1jey', '7axz']
    ckpt_ids = [i for i in range(0,200000, 2000) if os.path.isfile(f'{ckpt_dir}/step_{i}.ckpt')]
    infer_batch(model_gen, data_gen, sn, pdb_ids, ckpt_ids, res_dir, baseline=True)
    
    print(f'finish! {datetime.datetime.now()}')
    
    for i in range(120):
        time.sleep(60)
        i += 1
        print(f'sleep {i} minute(s)')
        
        
        
