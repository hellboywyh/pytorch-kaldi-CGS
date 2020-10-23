import os
import numpy as np

import torch
import sys
import config_train as cfg

def pruning(model, sparse_mode='dense'):
    if sparse_mode == 'dense':
        return model

    elif sparse_mode == 'thre_pruning':
        name_list = list()
        para_list = list()
        for name, para in model.named_parameters():
            name_list.append(name)
            para_list.append(para)

        a = model.state_dict()
        zero_cnt = 0
        all_cnt = 0
        for i, name in enumerate(name_list):
            raw_w = para_list[i]
            # raw_w.topk()
            zero = torch.zeros_like(raw_w)
            if name.split(".")[-2] != "bn" and name.split(".")[-1] != "bias":
                p_w = torch.where(abs(raw_w) < cfg.pruning_thre, zero, raw_w)
                zero_cnt += torch.nonzero(p_w).size()[0]
                all_cnt += torch.nonzero(raw_w).size()[0]
                a[name] = p_w
            else:
                a[name] = raw_w
        model.load_state_dict(a)

    elif sparse_mode == 'sparse_pruning':
        name_list = list()
        para_list = list()
        for name, para in model.named_parameters():
            name_list.append(name)
            para_list.append(para)

        a = model.state_dict()
        zero_cnt = 0
        all_cnt = 0
        for i, name in enumerate(name_list):
            raw_w = para_list[i]
            w_num = torch.nonzero(raw_w).size(0)
            zero_num = int(w_num * cfg.sparsity)
            if name.split(".")[-2] != "bn" and name.split(".")[-1] != "bias":
                value, _ = torch.topk(raw_w.abs().flatten(), w_num - zero_num)
                thre = abs(value[-1])
                zero = torch.zeros_like(raw_w)
                p_w = torch.where(abs(raw_w) < thre, zero, raw_w)
            
                zero_cnt += torch.nonzero(p_w).size()[0]
                all_cnt += torch.nonzero(raw_w).size()[0]
                a[name] = p_w
            else:
                a[name] = raw_w
            
        model.load_state_dict(a)

    elif sparse_mode == 'pattern_pruning':
        name_list = list()
        para_list = list()

        for name, para in model.named_parameters():
            name_list.append(name)
            para_list.append(para)

        a = model.state_dict()
        zero_cnt = 0
        all_cnt = 0
        for i, name in enumerate(name_list):
            raw_w = para_list[i]
            w_num = torch.nonzero(raw_w).size(0)
        
            # apply the patterns
            # mask = torch.tensor(cfg.pattern_mask[name])
            mask = cfg.pattern_mask[name].clone().detach()
            print(mask)
            p_w = raw_w * mask
            a[name] = p_w
        model.load_state_dict(a)

    elif sparse_mode == 'coo_pruning':
        name_list = list()
        para_list = list()
        pattern_shape  = cfg.coo_shape
        coo_nnz = cfg.coo_nnz 

        for name, para in model.named_parameters():
            name_list.append(name)
            para_list.append(para)

        a = model.state_dict()
        zero_cnt = 0
        all_cnt = 0
        for i, name in enumerate(name_list):
            raw_w = para_list[i]
            w_num = torch.nonzero(raw_w).size(0)
            
            # apply the patterns
            mask = torch.zeros_like(raw_w)
            if name.split(".")[-2] != "bn" and name.split(".")[-1] != "bias":
                # print(name, raw_w.size(), pattern_shape)
                if raw_w.size(0) % pattern_shape[0] == 0 and raw_w.size(1) % pattern_shape[1] == 0:
                    for k in range(raw_w.size(2)):
                        assert raw_w.size(0) % pattern_shape[0] == 0, f'{raw_w.size(0)} {pattern_shape[0]}'
                        for ic_p in range(raw_w.size(0) // pattern_shape[0]):
                            assert raw_w.size(1) % pattern_shape[1] == 0, f'{raw_w.size(1)} {pattern_shape[1]}'
                            for oc_p in range(raw_w.size(1) // pattern_shape[1]):
                                part_w = raw_w[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                                    oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k] 
                                value, _ = torch.topk(part_w.abs().flatten(), coo_nnz)
                                thre = abs(value[-1])
                                zero = torch.zeros_like(part_w)
                                one = torch.ones_like(part_w)
                                part_mask = torch.where(abs(part_w) < thre, zero, one)
                                mask[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                                oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k] = part_mask

                    p_w = raw_w * mask
                    zero_cnt += torch.nonzero(p_w).size()[0]
                    all_cnt += torch.nonzero(raw_w).size()[0]            
                    a[name] = p_w
                else:
                    a[name] = raw_w  
            else:
                a[name] = raw_w  

        model.load_state_dict(a)
        
    elif sparse_mode == 'ptcoo_pruning':
        name_list = list()
        para_list = list()
        pattern_shape  = cfg.pattern_shape
        pt_nnz  = cfg.pt_nnz
        coo_nnz = cfg.coo_nnz 

        for name, para in model.named_parameters():
            name_list.append(name)
            para_list.append(para)

        a = model.state_dict()
        zero_cnt = 0
        all_cnt = 0
        for i, name in enumerate(name_list):            
            raw_w = para_list[i]
            w_num = torch.nonzero(raw_w).size(0)
        
            # apply the patterns
            # mask = torch.tensor(cfg.pattern_mask[name])
            mask = cfg.pattern_mask[name].clone().detach()
            not_mask = torch.ones_like(cfg.pattern_mask[name]) - mask
            not_p_w = raw_w * not_mask


            raw_w = para_list[i]
            w_num = torch.nonzero(raw_w).size(0)
            
            # apply the patterns
            # mask = torch.zeros_like(raw_w)
            if name.split(".")[-2] != "bn" and name.split(".")[-1] != "bias":
                # print(name, raw_w.size(), pattern_shape)
                if raw_w.size(0) % pattern_shape[0] == 0 and raw_w.size(1) % pattern_shape[1] == 0:
                    for k in range(raw_w.size(2)):
                        assert raw_w.size(0) % pattern_shape[0] == 0, f'{raw_w.size(0)} {pattern_shape[0]}'
                        for ic_p in range(raw_w.size(0) // pattern_shape[0]):
                            assert raw_w.size(1) % pattern_shape[1] == 0, f'{raw_w.size(1)} {pattern_shape[1]}'
                            for oc_p in range(raw_w.size(1) // pattern_shape[1]):
                                not_part_w = not_p_w[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                                    oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k] 
                                value, _ = torch.topk(not_part_w.abs().flatten(), coo_nnz)
                                thre = abs(value[-1])
                                zero = torch.zeros_like(not_part_w)
                                one = torch.ones_like(not_part_w)
                                part_mask = torch.where(abs(not_part_w) < thre, zero, one)
                                mask[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                                oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k] += part_mask


                    p_w = raw_w * mask
                    zero_cnt += torch.nonzero(p_w).size()[0]
                    all_cnt += torch.nonzero(raw_w).size()[0]            
                    a[name] = p_w
                else:
                    a[name] = raw_w  
            else:
                a[name] = raw_w  

        model.load_state_dict(a)
        
    else:
        assert(False, "sparse mode does not exist")


    return model

def generate_pattern(pattern_num, pattern_shape, pattern_nnz):
    # generate the patterns
    patterns = torch.zeros([pattern_num, pattern_shape[0], pattern_shape[1]])
    for i in range(pattern_num):
        for j in range(pattern_nnz):
            random_row = np.random.randint(0, pattern_shape[0])
            random_col = np.random.randint(0, pattern_shape[1])
            # print(j, patterns[i, :, :])
            while patterns[i, random_row, random_col] == 1:
                random_row = np.random.randint(0, pattern_shape[0])
                random_col = np.random.randint(0, pattern_shape[1])
            patterns[i, random_row, random_col] = 1
        # print(patterns[i, :, :])
    return patterns

def generate_pattern_mask(model, patterns):
    name_list = list()
    para_list = list()
    patterns_mask = dict()
    pattern_shape = [patterns.size(1), patterns.size(2)]
    pattern_num = patterns.size(0)


    for name, para in model.named_parameters():
        name_list.append(name)
        para_list.append(para)

    a = model.state_dict()
    for i, name in enumerate(name_list):
        raw_w = para_list[i]
        w_num = torch.nonzero(raw_w).size(0)
        
        mask = torch.zeros_like(raw_w)
        if name.split(".")[-2] != "bn" and name.split(".")[-1] != "bias":
            if raw_w.size(0) % pattern_shape[0] == 0 and raw_w.size(1) % pattern_shape[1] == 0:
                for k in range(raw_w.size(2)):
                    assert raw_w.size(0) % pattern_shape[0] == 0, f'{raw_w.size(0)} {pattern_shape[0]}'
                    for ic_p in range(raw_w.size(0) // pattern_shape[0]):
                        assert raw_w.size(1) % pattern_shape[1] == 0, f'{raw_w.size(1)} {pattern_shape[1]}'
                        for oc_p in range(raw_w.size(1) // pattern_shape[1]):
                            
                            mask[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                                oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k] = cfg.patterns[np.random.randint(0, pattern_num), :, :]

                patterns_mask[name] = mask

            else:
                patterns_mask[name] = torch.ones_like(raw_w)
        else:
            patterns_mask[name] = torch.ones_like(raw_w)

    # pattern_test = find_pattern_layer(patterns_mask[name], pattern_shape)
    # print(pattern_test.values())
    # print(len(pattern_test.values()))
    # exit()
    return patterns_mask

def generate_pattern_mask_layerwise(model, pattern_num, pattern_shape, pattern_nnz): 
    name_list = list()
    para_list = list()
    patterns_mask = dict()
    patterns = generate_pattern(pattern_num, pattern_shape, pattern_nnz)

    for name, para in model.named_parameters():
        name_list.append(name)
        para_list.append(para)

    a = model.state_dict()
    for i, name in enumerate(name_list):

        patterns = generate_pattern(pattern_num, pattern_shape, pattern_nnz)
        raw_w = para_list[i]
        w_num = torch.nonzero(raw_w).size(0)
        
        mask = torch.zeros_like(raw_w)
        if name.split(".")[-2] != "bn" and name.split(".")[-1] != "bias":
            if raw_w.size(0) % pattern_shape[0] == 0 and raw_w.size(1) % pattern_shape[1] == 0:
                for k in range(raw_w.size(2)):
                    assert raw_w.size(0) % pattern_shape[0] == 0, f'{raw_w.size(0)} {pattern_shape[0]}'
                    for ic_p in range(raw_w.size(0) // pattern_shape[0]):
                        assert raw_w.size(1) % pattern_shape[1] == 0, f'{raw_w.size(1)} {pattern_shape[1]}'
                        for oc_p in range(raw_w.size(1) // pattern_shape[1]):
                            
                            mask[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                                oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k] = patterns[np.random.randint(0, pattern_num), :, :]

                patterns_mask[name] = mask

            else:
                patterns_mask[name] = torch.ones_like(raw_w)
        else:
            patterns_mask[name] = torch.ones_like(raw_w)

    # pattern_test = find_pattern_layer(patterns_mask[name], pattern_shape)
    # print(pattern_test.values())
    # print(len(pattern_test.values()))
    # exit()
    return patterns_mask

def find_pattern_certain_nnz_model(model, pattern_num, pattern_shape, pattern_nnz, if_pattern_prun=False):
    
    # pattern_num = 16
    # pattern_shape = [16, 16]
    # pattern_nnz = 32
    sparsity = pattern_nnz / (pattern_shape[0] * pattern_shape[1])
    patterns = dict()

    name_list = list()
    para_list = list()

    for name, para in model.named_parameters():
        if not para.dim() == 1:
            name_list.append(name)
            para_list.append(para)
            print(name, para.size())

    a = model.state_dict()
    zero_cnt = 0
    all_cnt = 0
    for i, name in enumerate(name_list):
        raw_w = para_list[i]
        raw_w, patterns_layer = find_pattern_certain_nnz_layer(raw_w, pattern_num, pattern_shape, pattern_nnz, if_pattern_prun)
        patterns = add_dict(patterns, patterns_layer)
        if if_pattern_prun:
            a[name] = raw_w.squeeze(2)
    model.load_state_dict(a)

    return model, patterns

def find_pattern_certain_nnz_layer(raw_w, pattern_num, pattern_shape, pattern_nnz, if_pattern_prun=False):
    patterns = dict()
    if raw_w.dim() == 2:
        raw_w = raw_w.unsqueeze(2)
    if not raw_w.size(0) % pattern_shape[0] == 0 or not raw_w.size(1) % pattern_shape[1] == 0:
        f"Error shape{raw_w.shape()}"
    mask = torch.ones_like(raw_w)
    for k in range(raw_w.size(2)):
        for ic_p in range(raw_w.size(0)// pattern_shape[0]):
            for oc_p in range(raw_w.size(1) // pattern_shape[1]):
                part_w = raw_w[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                            oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k]
                value, _ = torch.topk(part_w.abs().flatten(), pattern_nnz)

                # pruning
                thre = abs(value[-1])
                one = torch.ones_like(part_w)
                zero = torch.zeros_like(part_w)
                mask_p = torch.where(abs(part_w) < thre, zero, one)

                mask[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                            oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k] = mask_p

                # save the pattern
    patterns = find_pattern_layer(mask, pattern_shape)
    if if_pattern_prun:
        sorted_patterns = sorted(patterns.keys(),key=lambda item:patterns[item],reverse=True)
        selected_pattern_list = sorted_patterns[:pattern_num]
        raw_w = pattern_prun_certain_nnz_layer(raw_w, selected_pattern_list, pattern_shape)
    return raw_w, patterns
    

def pattern_prun_certain_nnz_layer(raw_w, selected_pattern_list, pattern_shape):
    if raw_w.dim() == 2:
        raw_w = raw_w.unsqueeze(2)
    selected_pattern_list = [torch.from_numpy(np.fromstring(selected_pattern_list[i],dtype=np.float32)).cuda().reshape(pattern_shape) for i in range(len(selected_pattern_list))]
    for k in range(raw_w.size(2)):
        for ic_p in range(raw_w.size(0)// pattern_shape[0]):
            for oc_p in range(raw_w.size(1) // pattern_shape[1]):
                part_w = raw_w[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                            oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k].abs()
                pattern_index = max(range(len(selected_pattern_list)),key=lambda i:torch.sum(selected_pattern_list[i]*(part_w)))
                raw_w[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                            oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k] = selected_pattern_list[pattern_index]
    return raw_w


def find_pattern_model(model, pattern_shape):
    
    patterns = dict()

    name_list = list()
    para_list = list()

    for name, para in model.named_parameters():
        name_list.append(name)
        para_list.append(para)

    for i, name in enumerate(name_list):
        if name.split(".")[-2] != "bn" and name.split(".")[-1] != "bias":
            raw_w = para_list[i]
            new_patterns = find_pattern_layer(raw_w, pattern_shape)
            patterns = add_dict(patterns, new_patterns)

    return patterns

def find_pattern_layer(raw_w, pattern_shape):
    
    patterns = dict()
    if raw_w.dim() == 2:
        raw_w = raw_w.unsqueeze(2)
    if raw_w.size(0) % pattern_shape[0] == 0 and raw_w.size(1) % pattern_shape[1] == 0:
        for k in range(raw_w.size(2)):
            for ic_p in range(int(raw_w.size(0)/ pattern_shape[0])):
                for oc_p in range(int(raw_w.size(1) / pattern_shape[1])):
                    part_w = raw_w[ic_p * pattern_shape[0]:(ic_p+1) * pattern_shape[0],
                                        oc_p * pattern_shape[1]:(oc_p+1) * pattern_shape[1], k]
                    zero = torch.zeros_like(part_w)
                    one = torch.ones_like(part_w)
                    pattern = torch.where(part_w == 0, zero, one).cpu().numpy().tostring()
                    # pattern.squeeze(dim=0)
                    if part_w.size(0) == pattern_shape[0] and part_w.size(1) == pattern_shape[1]:
                        if pattern not in patterns.keys():
                            patterns[pattern] = 1
                        else:
                            patterns[pattern] += 1
    return patterns

def add_dict(x, y):
    for k,v in x.items():
        if k in y.keys():
            y[k] += v
        else:
            y[k] = v
    return y

def cal_sparsity(model):        
    name_list = list()
    para_list = list()
    for name, para in model.named_parameters():
        name_list.append(name)
        para_list.append(para)

    zero_cnt = 0
    all_cnt = 0
    for i, name in enumerate(name_list):
        w = para_list[i]
        if name.split(".")[-2] != "bn":
            zero_cnt += w.flatten().size()[0] - torch.nonzero(w).size()[0]
            all_cnt += w.flatten().size()[0]

    return zero_cnt/all_cnt
