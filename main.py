# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""eval script"""
import argparse
import os
import datetime
import math
import stat
import json
import time
import ast
import psutil
import numpy as np
import pickle

import psutil
from concurrent.futures import ThreadPoolExecutor


np.set_printoptions(threshold=np.inf)

import mindspore.context as context
import mindspore.communication.management as D
from mindspore import Parameter
from mindspore.context import ParallelMode
from mindspore import Tensor, nn, save_checkpoint, load_checkpoint, load_param_into_net
from mindsponge1.cell.amp import amp_convert
from mindsponge1.cell.mask import LayerNormProcess
from mindsponge1.common.config_load import load_config

from data.dataset import create_dataset
from model import MegaFold
from module.fold_wrapcell import TrainOneStepCell, WithLossCell

from module.lr import cos_decay_lr
from data.permutation import placeholder_data_genenrator, ground_truth_generator
# from exp_info_sample import generate_interface_and_restraints
from restraint_sample import generate_interface_and_restraints

print(datetime.datetime.now(), 'Start now!')

parser = argparse.ArgumentParser(description='Inputs for eval.py')
parser.add_argument('--train_url', required=False, default="/job/output/", help='Location of training output')
parser.add_argument('--data_url', required=False, default="/job/dataset/", help='Location of data')
parser.add_argument('--data_config', default="/job/file/config/multimer-data.yaml", help='data process config')
parser.add_argument('--model_config', default="/job/file/config/multimer-model.yaml", help='model config')
parser.add_argument('--input_path', default="./examples/pkl/", help='processed raw feature path')
parser.add_argument('--pdb_path', default="./examples/pdb/", type=str, help='Location of training pdb file.')
parser.add_argument('--use_pkl', type=ast.literal_eval, default=True,
                    help="use pkl as input or fasta file as input, in default use fasta")
parser.add_argument('--checkpoint_path',type=str, default='jax_ckpt/converted_ms_ckpts/params_model_1_multimer_v3_ms.ckpt',  help='checkpoint path')
parser.add_argument('--device_id', default=0, type=int, help='DEVICE_ID')
parser.add_argument('--is_training', type=ast.literal_eval, default=True, help='is training or not')
parser.add_argument('--run_platform', default='Ascend', type=str, help='which platform to use, Ascend or GPU')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=True, help='run distribute')
parser.add_argument('--resolution_data', type=str, default=None, help='Location of resolution data file.')
parser.add_argument('--loss_scale', type=float, default=1024.0, help='loss scale')
parser.add_argument('--lr_max', type=float, default=0.0001, help='lr_max')
parser.add_argument('--gradient_clip', type=float, default=1.0, help='gradient clip value')
parser.add_argument('--total_steps', type=int, default=9600000, help='total steps')
parser.add_argument('--start_step', type=int, default=0, help='start step')
parser.add_argument('--extra_evonum', type=int, default=4, help='extra evoformer layer num') #4
parser.add_argument('--evonum', type=int, default=48, help='evoformer num') #48
parser.add_argument('--struct_recycle', type=int, default=8 , help='structure module num') # 8
parser.add_argument('--save_ckpt_path', type=str, required=True, help='save_ckpt_path')
parser.add_argument('--fix_afm', type=ast.literal_eval, default=False, help='fix parameters of AFM part')
parser.add_argument('--hard_rate', type=float, default=0.0, help='hard target rate')
parser.add_argument('--high', type=int, default=25, help='upper bound for number of msa in hard case')
parser.add_argument('--megafold', type=int, default=0, help='is megafold')

args = parser.parse_args()

pdb_path = f"{args.data_url}/pdb_all/"
pkl_path = f"{args.data_url}/pkl_all/"
paired_pkl_path = f"{args.data_url}/paired_msa/"

TARGET_FEAT_DIM = 21 #22
MSA_HEAD = 23 if args.megafold else 22


if args.checkpoint_path:
    load_ckpt_path = f"{args.train_url}/ckpt_dir/{args.checkpoint_path}"
else:
    load_ckpt_path = None

print(load_ckpt_path)

save_ckpt_path =f"{args.train_url}/ckpt_dir/{args.save_ckpt_path}/"
if not os.path.exists(save_ckpt_path):
    os.makedirs(save_ckpt_path, exist_ok=True)



os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
print(args)
print(os.environ)


def memory_stats():
    max_used_memory = 0
    tm0 = time.time()
    while True:
        tm = time.time() - tm0
        mem = psutil.virtual_memory()
        all_memory = int(mem.total) >> 30
        used_memory = int(mem.used) >> 30
        free_memory = int(mem.free) >> 30
        available_memory = int(mem.available) >> 30
        max_used_memory =int(max(max_used_memory, used_memory))
        tm = round(tm / 60.0, 2)
        storage_info = os.statvfs("./")
        free_size = storage_info.f_bsize * storage_info.f_bavail / (1024**3)

        CPU_use = psutil.cpu_percent(interval=1, percpu=True)
        CPU_use = np.mean(CPU_use)
        info = (f"time: {tm} min, all_memory: {all_memory}, used_memory: {used_memory}, free_memory: {free_memory}, available_memory: {available_memory}, "
                f"max_used_memory: {max_used_memory}, free storage: {free_size}, CPU_use: {CPU_use}")
        print(info, flush=True)

        time.sleep(60)

def random_sampling(names, prob_list, total):
    names_all = []
    while len(names_all) < total:
        random_prob = np.random.random(len(names))
        random_names = [names[i] for i in range(len(names)) if random_prob[i]<=prob_list[i]]
        names_all.extend(random_names)
    return names_all[ :total]

def loadjson(jsonfile):
    with open(jsonfile, 'r') as f:
        cont = json.load(f)
    return cont

def make_name_list(total_steps, start_step, seed=0, single_ratio=0.2):
    np.random.seed(seed)
    cluster_name_prob_path = f"{args.data_url}/namelist/cluster_name_prob_306.json"
    multimer_neighbor_namelist_path = f"{args.data_url}/namelist/multimer_neibor_namelist_0530.json"
    stop_date = "2020-05-01"
    print("stop_date", stop_date, flush=True)
    prob_name_all = loadjson(cluster_name_prob_path)
    resolution_data = {key.split("_")[0]: val["resolution"] for key, val in prob_name_all.items()}
    prob_name_all = {k: v for k, v in prob_name_all.items() if v["release_date"] > stop_date}
    multimer_namelist = loadjson(multimer_neighbor_namelist_path)
    names = [i for i in prob_name_all.keys() if len(multimer_namelist.get(i, ''))>1 and len(multimer_namelist.get(i, ''))<=2]
    print("num cases:", len(names))
    prob_list = [float(prob_name_all[x]['w_seqlen'])*float(prob_name_all[x]['w_clu_size_60w']) for x in names]
    namelist = random_sampling(names, prob_list, int(total_steps*(1-single_ratio))+1)

    single_names = [i for i in prob_name_all.keys() if len(multimer_namelist.get(i, ''))==1]
    prob_list_single = [float(prob_name_all[x]['w_seqlen'])*float(prob_name_all[x]['w_clu_size_60w']) for x in single_names]
    namelist.extend(random_sampling(single_names, prob_list_single, int(total_steps*single_ratio)+1))

    np.random.shuffle(namelist)
    mixed_names = [multimer_namelist[k] for k in namelist]
    divide_num = int(len(mixed_names)/device_num)
    t_names_all = []
    for x in range(device_num):
        t_names_all.append(mixed_names[x*divide_num:(x+1)*divide_num])
    node_id = int(rank/8)
    random_names_all = []
    for y in range(8):
        random_names_all.extend(t_names_all[node_id*8+y])
    random_names_all = random_names_all[:int(total_steps/(math.ceil(device_num/8.0)))]
    random_names_all = random_names_all[int(start_step*8):]
    return random_names_all, resolution_data

def main(args):
    data_cfg = load_config(args.data_config)
    data_cfg.common.max_extra_msa = 1024
    data_cfg.common.max_msa_clusters = 256
    data_cfg.common.crop_size = 384
    

    model_cfg = load_config(args.model_config)
    model_cfg.is_training = True
    model_cfg.seq_length = data_cfg.common.crop_size

    model_cfg.evoformer.msa_stack_num = args.evonum
    model_cfg.evoformer.extra_msa_stack_num = args.extra_evonum
    model_cfg.structure_module.num_layer = args.struct_recycle

    slice_key = "seq_" + str(model_cfg.seq_length)
    slice_val = vars(model_cfg.slice)[slice_key]
    model_cfg.slice = slice_val

    data_cfg.common.target_feat_dim = TARGET_FEAT_DIM
    model_cfg.common.target_feat_dim = TARGET_FEAT_DIM

    model_cfg.heads.masked_msa.num_output = MSA_HEAD

    np.random.seed(0) #
    megafold = MegaFold(model_cfg, mixed_precision=args.mixed_precision)
    fp32_white_list = (nn.Softmax, nn.LayerNorm, LayerNormProcess)
    amp_convert(megafold, fp32_white_list)

    net_with_criterion = WithLossCell(megafold, model_cfg)
    lr_total_steps = args.total_steps // device_num #10000
    print('total step:',  lr_total_steps)
    
    # lr = cos_decay_lr(start_step=args.start_step, lr_init=0.0,
    #                     lr_min=0.00005, lr_max=0.0001, decay_steps=30000,
    #                     total_steps=lr_total_steps, 
    #                     warmup_steps=0)
    lr = cos_decay_lr(start_step=args.start_step, lr_init=0.0,
                        lr_min=0.0001, lr_max=0.0001, decay_steps=30000,
                        total_steps=lr_total_steps, 
                        warmup_steps=0)
    print('learning rate:', lr[:20])
    if args.fix_afm:
        params_list = []
        for k, v in megafold.parameters_and_names():
            if ('sbr' in k) or ('interface' in k):
                params_list.append(v)
        opt = nn.Adam(params=params_list, learning_rate=lr, eps=1e-6)
    else:
        opt = nn.Adam(params=megafold.trainable_params(), learning_rate=lr, eps=1e-6)
    train_net = TrainOneStepCell(net_with_criterion, opt, sens=args.loss_scale,
                                 gradient_clip_value=args.gradient_clip)
    
    if load_ckpt_path:
        print("load_checkpoint(load_ckpt_path, train_net)", load_ckpt_path)
        
        param_dict_ori = load_checkpoint(load_ckpt_path)
        keys = list(param_dict_ori.keys())
        for key in keys:
            if "learning_rate" in key or "global_step" in key:
                param_dict_ori.pop(key)

            if (('preprocess_1d.weight' in key) or ('left_single.weight' in key) or ('right_single.weight' in key)) and ('megafold' in load_ckpt_path) and (param_dict_ori[key].shape[-1] == 22):
                param_dict_ori[key] = Parameter(param_dict_ori[key][..., 1:])
        # for key in keys:
        #     if "learning_rate" in key or "global_step" in key or "moment1" in key or "moment2" in key or "beta1_power" in key or "beta2_power" in key:
        #         param_dict_ori.pop(key)
        # # continue
        load_param_into_net(train_net, param_dict_ori)

        # load_checkpoint(load_ckpt_path, train_net)

    train_net.set_train(False)
    step = args.start_step
    max_recycles = [int(np.random.uniform(size=1, low=1, high=5)) for _ in range(args.total_steps)]
    max_recycles[step] = 1
    
    all_name_list, resolution_data = make_name_list(args.total_steps, args.start_step)
    print(all_name_list[:10])

    train_dataset = create_dataset(pdb_path, pkl_path, paired_pkl_path, all_name_list, data_cfg,
                                   resolution_data, num_parallel_worker=8, is_parallel=True,
                                   shuffle=False, mixed_precision=args.mixed_precision,
                                   hard_rate=args.hard_rate, high=args.high)
    dataset_iter = train_dataset.create_dict_iterator(num_epochs=1, output_numpy=True)


    np.random.seed(rank*1000+1)
    for d in dataset_iter:

        prot_name = all_name_list[int(d["prot_name_index"][0])]

        if step % 1000 == 0 and rank == 0:
            ckpt_name = save_ckpt_path + '/' + f"step_{step}.ckpt"
            save_checkpoint(train_net, ckpt_name)

        # if step == 0 and rank % 8 == 0:
        #     pdb_id = '_and_'.join(prot_name)
        #     with open(f'{save_ckpt_path}/{pdb_id}_feat.pkl', 'wb') as f:
        #         pickle.dump(d, f)

        info = f"rank: {rank}, step: {step}, prot_name {prot_name} {datetime.datetime.now()}"
        print(info, flush=True)
        print(step,  datetime.datetime.now(), prot_name, flush=True)
        # show_npdict(d, "dataset_features")

        sbr, sbr_mask, interface_mask = generate_interface_and_restraints(d, fix_afm=args.fix_afm)

        # keys = ["aatype", "residue_index", "template_aatype", "template_all_atom_masks", "template_all_atom_positions", "asym_id", "sym_id", "entity_id", \
        #         "seq_mask", "msa_mask", "target_feat", "msa_feat", "extra_msa", "extra_msa_deletion_value", "extra_msa_mask", "residx_atom37_to_atom14", "atom37_atom_exists"]

        max_recycle = max_recycles[step]
        
        inputs_feat0 = d["aatype"], d["residue_index"], d["template_aatype"], \
                       d["template_all_atom_masks"], d["template_all_atom_positions"], \
                       d["asym_id"], d["sym_id"], d["entity_id"], \
                       d["seq_mask"], d["msa_mask"], d["target_feat"], \
                       d["msa_feat"], d["extra_msa"], d["extra_msa_deletion_value"], \
                       d["extra_msa_mask"], d["residx_atom37_to_atom14"], d["atom37_atom_exists"]
                       
        
        prev_pos, prev_msa_first_row, prev_pair = Tensor(d["prev_pos"]), Tensor(d["prev_msa_first_row"]), \
                                                  Tensor(d["prev_pair"])
        ground_truth_fake = placeholder_data_genenrator(num_res=d["msa_mask"].shape[2],
                                        num_msa=d["msa_mask"].shape[1])
        if max_recycle == 1:
            train_prev_pos = prev_pos
            train_prev_msa_first_row = prev_msa_first_row
            train_prev_pair = prev_pair

        # forward recycle 3 steps
        train_net.add_flags_recursive(train_backward=False)
        train_net.phase = 'train_forward'
        ground_truth_fake = [Tensor(gt) for gt in ground_truth_fake]
        print("tm0", datetime.datetime.now(), flush=True)
        for recycle in range(max_recycle):
            inputs_feat = [feat[recycle] for feat in inputs_feat0]
            inputs_feat = inputs_feat + [sbr, sbr_mask, interface_mask]
            inputs_feat = [Tensor(i) for i in inputs_feat]
            prev_pos, prev_msa_first_row, prev_pair, _, _, _ = train_net(*inputs_feat,
                                                                   prev_pos, prev_msa_first_row,
                                                                   prev_pair, *ground_truth_fake)
            if recycle == (max_recycle - 2):
                train_prev_pos = prev_pos
                train_prev_msa_first_row = prev_msa_first_row
                train_prev_pair = prev_pair
    

        print("tm1", datetime.datetime.now(), flush=True)
        ground_truth = ground_truth_generator(d, prev_pos.asnumpy(), max_recycle)

        print("tm2", datetime.datetime.now(), flush=True)
        ground_truth = [Tensor(feat) for feat in ground_truth]
        inputs_feat = [feat[max_recycle-1] for feat in inputs_feat0]
        inputs_feat = inputs_feat + [sbr, sbr_mask, interface_mask]
        inputs_feat = [Tensor(i) for i in inputs_feat]
        # forward + backward
        train_net.add_flags_recursive(train_backward=True)
        train_net.phase = 'train_backward'
        loss = train_net(*inputs_feat, train_prev_pos, train_prev_msa_first_row, train_prev_pair, *ground_truth)

        tm3 = time.time()
        # print("backward cost", tm3- tm2)
        # print("all time", tm3- tm0)

        loss_name = ['total_loss', 'l_fape_side', 'l_fape_backbone', 'l_anglenorm', 'distogram_loss', 'masked_loss', 
                     'predict_lddt_loss', 'structure_violation_loss', 'no_clamp', ' fape_nc_intra', 'fape_nc_inter', 
                     'chain_centre_mass_loss', 'aligned_error_loss', 'sbr_inter_fape_loss', 'sbr_inter_drmsd_loss', 
                     'sbr_inter_disto_loss', 'sbr_intra_fape_loss', 'sbr_intra_drmsd_loss', 'sbr_intra_disto_loss', 
                     'interface_loss', 'recall_intra', 'recall_inter', 'recall_interface', 'perfect_recall_interface',
                     'recall_inter1', 'recall_intra1']
        loss_info = ', '.join([f'{k}: {v}' for k, v in zip(loss_name, loss)])
        loss_info = (f"step is: {step}===={loss_info}====time {datetime.datetime.now()}, prot {prot_name}")
        print(loss_info, flush=True)
        step += 1
        '''
        '''
        # if step == 5:
        #     break

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        max_device_memory="29GB",
                        max_call_depth=6000,
                        #device_id=args.device_id,
                        save_graphs=False,
                        save_graphs_path="/job/file/graphs/")
    D.init()
    device_num = D.get_group_size()
    rank = D.get_rank()
    device_id = int(os.getenv('DEVICE_ID'))
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                        device_num=device_num, parameter_broadcast=False)
    args.mixed_precision = 1
    if rank % 8 == 0:
        # if True:
        thread_pool_2 = ThreadPoolExecutor(max_workers=1)
        thread_pool_2.submit(memory_stats)
    main(args)
