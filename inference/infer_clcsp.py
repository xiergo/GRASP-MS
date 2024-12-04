import argparse
import os
import glob
import time
import datetime
import pickle
import numpy as np
from restraint_sample import BINS
from utils_infer import infer_config, infer_batch, DataGenerator, ModelGenerator


def switch_idx(old_idxs, seqs):
    print(f'before switch: {old_idxs}')
    def global2local(global_idx, seq_lens):
        # 0-based index 
        # 0 -> 0, 0
        
        seq_lens = [0]+list(seq_lens)
        seq_cum_lens = np.cumsum(seq_lens)
        cid = ((global_idx-seq_cum_lens)>=0).sum()-1
        local_idx = global_idx - seq_cum_lens[cid]
        return cid, local_idx
    
    seq_order = {}
    for cid, seq in enumerate(seqs):
        if seq not in seq_order:
            seq_order[seq] = []
        seq_order[seq].append(cid)
    
    cid_order_list = []
    seq_len_list = []
    for seq, cids in seq_order.items():
        for cid in cids:
            cid_order_list.append(cid)
            seq_len_list.append(len(seq))
    
    new_idxs = []
    for old_idx in old_idxs:
        old_cid, old_local_idx = global2local(old_idx, [len(seq) for seq in seqs])
        new_idx = sum(seq_len_list[:cid_order_list.index(old_cid)]) + int(old_local_idx)
        new_idxs.append(new_idx)
    print(f'after switch (before sort): {new_idxs}')
    new_idxs.sort()
    return new_idxs
    
def get_colabdock_mvn_interface(pkl_path, seqs):
    with open(pkl_path, 'rb') as f:
        pos = pickle.load(f)
    # pos1 = pos
    pos = np.concatenate([[j[0] for j in i[:-1]] for i in pos])-1 
    pos = list(pos)
    seqlen = sum([len(seq) for seq in seqs])
    pos = switch_idx(pos, seqs)
    bins = BINS
    sbr = np.zeros((seqlen, seqlen, len(bins) + 1))
    sbr_mask = np.zeros((seqlen, seqlen))
    print(f'{len(pos)} interfaces: {pos}')
    interface_mask = np.zeros(seqlen)
    interface_mask[pos] = 1
    restraints = {}
    restraints['sbr'] = sbr
    restraints['sbr_mask'] = sbr_mask
    restraints['interface_mask'] = interface_mask
    return restraints


class MyDataGenerator(DataGenerator):
    def __init__(self, raw_feat_dir, fasta_dir, restr_dir, reorder=False):
        super().__init__(raw_feat_dir, fasta_dir, reorder)
        self.restr_dir = restr_dir
    
    def get_data(self, pdb_id):
        raw_feature = self.get_feat(pdb_id)
        # seqlen = raw_feature['msa'].shape[1]

        seq_dict = self.get_seqs_dict(pdb_id)
        print(seq_dict)
        seqs = list(seq_dict.values())


        restr_file = glob.glob(f'{self.restr_dir}/{pdb_id}*pkl')[0]
        restraints = get_colabdock_mvn_interface(restr_file, seqs)
        restraints['asym_id'] = raw_feature['asym_id']
        save_data=True
        if save_data:
            save_data_dir = f'/job/output/save_data/clcsp_data'
            os.makedirs(save_data_dir, exist_ok=True)
            with open(f'{save_data_dir}/{pdb_id}_restr.pkl', 'wb') as f:
                pickle.dump(restraints, f)
            with open(f'{save_data_dir}/{pdb_id}_raw_feat.pkl', 'wb') as f:
                pickle.dump(raw_feature, f)
            # import sys
            # sys.exit()
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
    parser.add_argument('--ckpt_ids', default=None, type=str)
    parser.add_argument('--jobname', default='clcsp_new', type=str)
    arguments = parser.parse_args()
    
    # set path
    raw_feat_dir = f'{arguments.data_url}/cl_csp/features'
    fasta_dir = f'{arguments.data_url}/cl_csp/fasta'
    restr_dir = f'{arguments.data_url}/cl_csp/restr'
    res_dir = f'{arguments.train_url}/grasp_infer_res/{arguments.ckpt_dir}_{arguments.jobname}'
    ckpt_dir = f'{arguments.train_url}/ckpt_dir/{arguments.ckpt_dir}'
    
    data_gen = MyDataGenerator(raw_feat_dir, fasta_dir, restr_dir)
    model_gen = ModelGenerator(arguments, ckpt_dir)
    sn = infer_config(rotate_split=False, outdir=res_dir)
    sn.start_job()
    
    # confs = ['interface', 'sbr', 'both']
    pdb_ids = [i.split('.')[0] for i in os.listdir(fasta_dir)]
    # pdb_ids = [i for i in pdb_ids if i.startswith('4INS')]
    # pdb_ids = [f'{i}_{j}' for i in pdb_ids for j in confs]

    if arguments.ckpt_ids is not None:
        ckpt_ids = [int(i) for i in arguments.ckpt_ids.split('-')]
    else:
        ckpt_ids = [i*1000 for i in [8, 14, 20, 22]]+[22946222]
    ckpt_ids.sort()
    infer_batch(model_gen, data_gen, sn, pdb_ids, ckpt_ids, res_dir)
    
    print(f'finish! {datetime.datetime.now()}')
    sn.complete()
    
    for i in range(3600):
        if sn.check_all_complete():
            break
        time.sleep(60)
        i += 1
        print(f'sleep {i} minute(s)')
        
        
        
