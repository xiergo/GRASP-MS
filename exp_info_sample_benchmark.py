import numpy as np

BINS = np.arange(8, 26, 2)

def get_crop_index(asym_id, residue_index, chain_index):
    unique_chain_index = np.unique(chain_index)
    unique_chain_index.sort()
    crop_index = []
    seq_len = 0
    for i in unique_chain_index:
        # print(i, seq_len)
        res_idx_i = residue_index[asym_id == (i+1)]
        res_idx_i += seq_len
        crop_index.extend(res_idx_i)
        seq_len += (chain_index == i).sum()
        # print(crop_index)
    return crop_index


def generate_mask(dist, pseudo_beta_mask, asym_id, residue_index):
    ''' add mask info'''
    remote_residue_threshold = 6
    greater_residue_index_diff = (np.abs(residue_index[:, None] - residue_index[None]) > remote_residue_threshold).astype(np.float32)
    pseudo_beta_mask_2d =  (pseudo_beta_mask[:,None] * pseudo_beta_mask[None]) * (dist > 0.01)
    upper_mask = np.triu(np.ones_like(pseudo_beta_mask_2d), 1)
    mask_intra = (asym_id[:, None] == asym_id[None]).astype(np.float32)

    mask_inter = (1.0 - mask_intra) * pseudo_beta_mask_2d
    mask_inter_contact = (dist < 8.0) * mask_inter
    mask_all_contact = (dist < 8.0) * pseudo_beta_mask_2d
    
    mask_inter *= upper_mask
    mask_intra = mask_intra * greater_residue_index_diff * pseudo_beta_mask_2d * upper_mask

    return mask_inter, mask_intra, mask_inter_contact, mask_all_contact

def get_sample_num(start, reduce, end, k):
    probs = np.concatenate((np.zeros(start), np.ones(reduce-start), np.exp(-(np.arange(end-reduce)/k))),axis=0)
    probs /= (probs.sum() + 1e-8)
    num = np.random.choice(np.arange(end), p=probs)
    return num

def freq_each_bin(dist, mask, bins):
    mask = (mask < 0.5)
    dist = dist - mask * 1e8
    bins = np.array(list(bins) + [1e8])
    mask_num = mask.sum()
    num_under_binedges = (dist[..., None] < bins).sum(axis=(-2, -3))
    num_in_bins = num_under_binedges[1:] - num_under_binedges[: -1]
    num_in_bins = np.array([num_under_binedges[0] - mask_num] + list(num_in_bins))
    return num_in_bins


def get_probs(d, mask=None, bins=BINS):
    if mask is None:
        mask = np.ones_like(d)
    if not mask.sum() > 0:
        return np.zeros_like(d)
    num_in_bins = freq_each_bin(d, mask, bins)
    probs_each_bin =  1.0 / (num_in_bins + 1e-6)
    d_onehot_index = (d[..., None] > bins).sum(-1)
    probs1 = probs_each_bin[d_onehot_index]
    # print(probs1)
    d0 = bins[0]
    d1 = bins[-1]
    probs2 = np.where(d < d0, 4.0, np.exp(-(d-d0)/(d1-d0)))
    # print(probs2)
    probs = probs1 * probs2 * mask
    if probs.sum()>0:
        probs /= probs.sum()
    else:
        probs = mask/mask.sum()
    return probs


def get_sample_mask_by_probs(probs, num):
    sample_mask = np.zeros(probs.size)
    non_zero = (probs>0).sum()
    num = min(non_zero, num)
    if num == 0:
        return sample_mask.reshape(probs.shape), num
    flatten_index = np.random.choice(np.arange(probs.size), size=num, p=probs.ravel(), replace=False)
    sample_mask[flatten_index] = 1
    sample_mask = sample_mask.reshape(probs.shape)
    return sample_mask, num


def show_dists(dist, mask, title = 'distances sampled:'):
    print(f'>>>>>>>>> {title}')
    if mask.sum() <= 0:
        print('No distance sampled')
    else:
        d = dist[mask > 0.5]
        print(f'num: {mask.sum()}, median: {np.median(d)}, mean: {d.mean()}')
        print(d[:20])
    print('>>>>>>>>>>>')
    

def sample_by_distance(dist, num, mask=None):
    assert dist.shape == mask.shape
    probs = get_probs(dist, mask=mask)
    sample_mask = get_sample_mask_by_probs(probs, num)
    return sample_mask
    
def onehot(d, bins):
    n = (d[..., None]>bins).sum(-1)
    return np.eye(len(bins)+1)[n]

def get_lastdim_distri(d, bins):
    d_onehot = onehot(d, bins)
    d_freq = d_onehot.sum(-2)
    d_perc = d_freq / (d_freq.sum(-1, keepdims=True) + 1e-8)
    return d_perc

def soft_blurred_retraints(d, bins):
    sigma = np.random.uniform(low=0, high=3, size=d.shape)
    mu = d + np.random.randn(*d.shape)
    # print(mu, sigma, sep='\n')
    d_sample = mu[..., None] + sigma[..., None] * np.random.randn(100)
    br = get_lastdim_distri(d_sample, bins)
    return br

def sample_soft_blurred_restraints(dist, sbr, mask, num):
    mask, num = sample_by_distance(dist, num, mask)
    sbr = sbr * mask[..., None]
    return sbr, num, mask

def sample_interface_by_asym_id(asym_id, mask, num):
    num = min(num, mask.sum())
    mask_interface = np.zeros_like(mask)
    if num > 0:
        asym_id_same = asym_id[:, None] == asym_id[None]
        num_interface_each_chain = (asym_id_same * mask[None]).sum(-1)
        probs = mask / (num_interface_each_chain + 1e-8)
        probs /= (probs.sum() + 1e-8)
        idx = np.random.choice(np.arange(len(asym_id)), num, replace=False, p=probs)
        mask_interface[idx] = 1.0
    return mask_interface, num
   

def generate_interface_and_restraints(d, mixed_precision=True, training=True,
                                      num_inter=None, num_intra=None, num_interface=None, 
                                      only_contact_sbr = False, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # assert 'pseudo_beta' in d
    # assert 'pseudo_beta_mask' in d
    # assert 'asym_id' in d
    # assert 'residue_index' in d
    # assert 'chain_index' in d
    
    if training:
        asym_id = d['asym_id'][0]    
        residue_index = d['residue_index'][0]
        chain_index = d['chain_index']
        crop_index = get_crop_index(asym_id, residue_index, chain_index)
        seqlen = len(asym_id) #384
        pseudo_beta = d["pseudo_beta"][crop_index]
        pseudo_beta_mask = d['pseudo_beta_mask'][crop_index]
        # pad to fixed length
        pseudo_beta = np.pad(pseudo_beta, ((0, seqlen - pseudo_beta.shape[0]), (0, 0)))
        pseudo_beta_mask = np.pad(pseudo_beta_mask, ((0, seqlen - pseudo_beta_mask.shape[0]),))
        dist = np.sqrt((np.square(pseudo_beta[None]-pseudo_beta[: ,None])).sum(-1) + 1e-8)

    else:
        asym_id = d['asym_id']
        seqlen = len(asym_id)
        pseudo_beta = d['pseudo_beta'] if 'pseudo_beta' in d else np.zeros((seqlen, 3))
        if 'pseudo_beta_mask' in d:
            pseudo_beta_mask = d['pseudo_beta_mask']
        elif 'mask_2d' in d:
            pseudo_beta_mask = (d['mask_2d'].sum(0) > 0.5).astype(d['mask_2d'].dtype)
        else:
            np.ones_like(asym_id)
        dist = np.sqrt((np.square(pseudo_beta[None]-pseudo_beta[: ,None])).sum(-1) + 1e-8)
        dist = d['dist'] if 'dist' in d else dist
        residue_index = d['residue_index'] if 'residue_index' in d else np.arange(seqlen)
     
    bins = BINS
    sbr = np.zeros((seqlen, seqlen, len(bins) + 1))
    sbr_inter = np.zeros((seqlen, seqlen, len(bins) + 1))
    sbr_intra = np.zeros((seqlen, seqlen, len(bins) + 1))
    sbr_mask = np.zeros((seqlen, seqlen))
    mask_interface = np.zeros(seqlen)
    
    try:
        if num_inter is None:
            num_inter = get_sample_num(start=2, reduce=20, end=40, k=4)
        if num_intra is None:
            num_intra = get_sample_num(start=16, reduce=80, end=512, k=32)
        if num_interface is None:
            num_interface = get_sample_num(start=10, reduce=30, end =60, k=4)

        mask_inter, mask_intra, mask_inter_contact, mask_all_contact = generate_mask(dist, pseudo_beta_mask, asym_id, residue_index)

        if only_contact_sbr:
            mask_intra = mask_intra * mask_all_contact
            mask_inter = mask_inter * mask_all_contact
            sbr[..., 0] = 1.0
        else:
            sbr = soft_blurred_retraints(dist, bins)

        sbr_intra, num_intra, mask_intra_chozen = sample_soft_blurred_restraints(dist, sbr, mask_intra, num_intra)
        show_dists(dist, mask_intra_chozen, 'distances of intra restrains:')

        sample_sbr_first = np.random.rand() < 0.5
        if sample_sbr_first:
            # sample inter-chain soft blurred restraints first
            sbr_inter, num_inter,  mask_inter_chozen = sample_soft_blurred_restraints(dist, sbr, mask_inter, num_inter)
            mask_inter_contact_exclude = (mask_inter_chozen + mask_inter_chozen.T) * mask_inter_contact
            mask_inter_contact = mask_inter_contact * (1 - mask_inter_contact_exclude)
            mask_interface = mask_inter_contact.any(axis=0)
            mask_interface, num_interface = sample_interface_by_asym_id(asym_id, mask_interface, num_interface)
        else:
            # sample interface first
            mask_interface = mask_inter_contact.any(axis=0)
            mask_interface, num_interface = sample_interface_by_asym_id(asym_id, mask_interface, num_interface)
            mask_inter *= (1 - mask_interface[:, None] * mask_interface[None])
            sbr_inter, num_inter, mask_inter_chozen = sample_soft_blurred_restraints(dist, sbr, mask_inter, num_inter)
        show_dists(dist, mask_inter_chozen, 'distances of inter restrains:')
        sbr = sbr_inter + sbr_intra
        sbr = sbr + sbr.swapaxes(0, 1)
    except Exception as e:
        print('Generate experimental information error!')
        print(e)
    
    dtype = np.float32
    if mixed_precision:
        dtype = np.float16
    sbr = sbr.astype(dtype)
    sbr_mask = (sbr.sum(-1) > 0.5).astype(dtype)
    mask_interface = mask_interface.astype(dtype)

    return sbr, sbr_mask, mask_interface, num_inter, num_intra, num_interface


# if __name__ == '__main__':
#     sbr, sbr_mask, mask_interface, num_inter, num_intra, num_interface = generate_interface_and_restraints(d)
#     print(sbr_mask.sum())
#     print(sbr.shape)
#     print(mask_interface)
#     print(num_inter, num_intra, num_interface)