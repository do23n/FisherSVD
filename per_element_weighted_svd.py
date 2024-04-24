import torch
import torch.nn as nn
from modules.svd_linear import SVDLinear


def get_low_rank(w, rank):
    """
    :param tensor w: matrix to decompose
    :param int rank: desired rank
    """
    u, s, vt = torch.linalg.svd(w, full_matrices=False)
    u = u[:, :rank]
    s = s[:rank]
    vt = vt[:rank]
    return u, s, vt

def per_element_weighted_svd(w, weight, rank, max_iter=10, threshold=1e-3):
    """
    :param tensor w: matrix to decompose
    :param tensor weight: weights of elements
    :param int rank: desired rank
    :param int max_iter: number of iterations
    """
    u, s, vt = get_low_rank(w, rank)
    s = torch.zeros(rank, device=s.device)
    prev_norm = (u @ torch.diag(s) @ vt).norm(p=2)

    if weight is None:
        return u, s, vt
    
    for i in range(max_iter):
        if i%5 == 0:
            print(f"iter {i}")
        u, s, vt = get_low_rank(weight * w + (1-weight) * (u @ torch.diag(s) @ vt), rank)
        cur_norm = (u @ torch.diag(s) @ vt).norm(p=2)
        if torch.abs(prev_norm - cur_norm) <= threshold:
            return u, s, vt
        prev_norm = cur_norm

    return u, s, vt

    
def decompose_model(model, sensitivity_dict, rank, max_iter, threshold, ratio = None):        
    print(f"Start decompose_model:")
    layer_names = list(sensitivity_dict.keys())
    if 'full_model' in layer_names:
        layer_names.remove('full_model')

    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    tot_params = 0
    compress_params = 0

    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    for layername in layer_names:
        print(f"layer: {layername}")
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        tot_params += raw_linear.weight.numel()
        if ratio:
            w_compress_params = int(raw_linear.weight.numel() * ratio)
            compress_params += w_compress_params
            rank = w_compress_params // (raw_linear.in_features + raw_linear.out_features)
        else:
            compress_params += (raw_linear.weight.shape[0] + raw_linear.weight.shape[1]) * rank
        print(f'raw shape: {raw_linear.weight.shape}')
        print(f'raw n_params: {raw_linear.weight.numel()}')
        if ratio:
            print(f'ratio compress: {int(tot_params * ratio)}')
            print(f'rank after ratio: {rank}')
        else:
            print(f' rank compress: {(raw_linear.weight.shape[0] + raw_linear.weight.shape[1]) * rank}')


        w = raw_linear.weight.data.float()
        weight = raw_linear.fisher_info_per_element
        u, s, vt = per_element_weighted_svd(w, weight, rank, max_iter, threshold)
        v = vt.T
        svd_linear = SVDLinear(u,s,v)
        setattr(info["father"], info["name"], svd_linear)

    compress_ratio = compress_params / tot_params
    return compress_ratio
    
