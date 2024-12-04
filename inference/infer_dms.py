import argparse
import os
import glob
import time
import datetime
import pandas as pd
import numpy as np
from restraint_sample import BINS
from utils_infer import infer_config, infer_batch, DataGenerator, ModelGenerator

def get_mapping_from_fasta(fasta):
    with open(fasta, 'r') as f:
        cont = [i.strip() for i in f.readlines()]
    mapping = {}
    for i in range(0, len(cont), 2):
        k = cont[i].split('_')[-1]
        v = len(cont[i+1])
        mapping[k] = v
    return mapping
 
def get_dms_interface(restr_file, mapping, thre=0.2, cheat_mode=0):
    df = pd.read_csv(restr_file, sep='\t')
    seqlen = sum(mapping.values())
    start_pos = sum(list(mapping.values())[:list(mapping.keys()).index('sp')])
    bins = BINS
    sbr = np.zeros((seqlen, seqlen, len(bins) + 1))
    sbr_mask = np.zeros((seqlen, seqlen))
    interface_mask = np.zeros(seqlen)
    print(df)
    print(f'total sites: {df.shape[0]}: {list(df.dist)}')
    df = df[df.max_es>=thre]
    print(f'after filtered by max_es>={thre}: {df.shape[0]}: {list(df.dist)}')
    if cheat_mode:
        df = df[df.dist<=8]
        print(f'after using cheat mode: {df.shape[0]}: {list(df.dist)}')
    for i in df.pos:
        interface_mask[i+start_pos-1] = 1
    restraints =  {'sbr': sbr, 'sbr_mask': sbr_mask, 'interface_mask': interface_mask}
    return restraints


class MyDataGenerator(DataGenerator):
    def __init__(self, raw_feat_dir, fasta_dir, restr_dir, reorder=False):
        super().__init__(raw_feat_dir, fasta_dir, reorder)
        self.restr_dir = restr_dir

    def get_pattern(self, pdb_id):
        pdb_id, _, _ = pdb_id.rsplit('_', 2)
        pat_dict = super().get_pattern(pdb_id)
        restr_pat = f'{self.restr_dir}/{pdb_id}*dms.tsv'
        pat_dict.update({'restr': restr_pat})
        return pat_dict
    
    def get_data(self, pdb_id):
        raw_feature = self.get_feat(pdb_id)
        _, cheat, thre = pdb_id.rsplit('_', 2)
        cheat = int(cheat.replace('cheat', ''))
        thre = float(thre.replace('thre', ''))
        mapping = get_mapping_from_fasta(self.get_files(pdb_id)['fasta'])
        restr_file = self.get_files(pdb_id)['restr']
        restraints = get_dms_interface(restr_file, mapping, thre=thre, cheat_mode=cheat)
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
    parser.add_argument('--jobname', default='dms', type=str)
    # parser.add_argument('--cheat_mode', default=0, type=int)
    # parser.add_argument('--thre', default=0.2, type=float)
    parser.add_argument('--ckpt_ids', default=None, type=str)

    
    arguments = parser.parse_args()
    print(arguments)
    
    # set path
    raw_feat_dir = f'{arguments.data_url}/dms_data_full_antigen/features'
    fasta_dir = f'{arguments.data_url}/dms_data_full_antigen/fasta'
    restr_dir = f'{arguments.data_url}/dms_data_full_antigen/dms'
    res_dir = f'{arguments.train_url}/grasp_infer_res/{arguments.ckpt_dir}_{arguments.jobname}'
    ckpt_dir = f'{arguments.train_url}/ckpt_dir/{arguments.ckpt_dir}'
    
    data_gen = MyDataGenerator(raw_feat_dir, fasta_dir, restr_dir, reorder=False)
    model_gen = ModelGenerator(arguments, ckpt_dir)
    sn = infer_config(rotate_split=False, outdir=res_dir)
    sn.start_job()
    
    pdb_ids = [i.split('.fasta')[0] for i in os.listdir(fasta_dir)]
    pdb_ids = [
        f'{pdb_id}_cheat{i}_thre{j}' \
        for pdb_id in pdb_ids \
        for i in [0, 1] \
        for j in [0, 0.2] \
        if not ((i == 1) and (j == 0.2))
    ]

    
    if arguments.ckpt_ids is not None:
        ckpt_ids = [int(i)*1000 for i in arguments.ckpt_ids.split('-')]
    else:
        ckpt_ids = [i*1000 for i in range(8, 27, 2)]
    ckpt_ids.sort()
    print('ckpt_ids', ckpt_ids)

    infer_batch(model_gen, data_gen, sn, pdb_ids, ckpt_ids, res_dir, baseline=False, num_seed=5, check_tsv_exist=True)
    
    print(f'finish! {datetime.datetime.now()}')
    sn.complete()

    for i in range(3600):
        if sn.check_all_complete():
            break
        time.sleep(60)
        i += 1
        print(f'sleep {i} minute(s)')
        
        
        
