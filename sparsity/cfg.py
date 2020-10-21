'''
Description: 
version: 
Author: Wang Yanhong
email: 284520535@qq.com
Date: 2020-10-21 07:20:07
LastEditors: Wang Yanhong
LastEditTime: 2020-10-21 07:20:41
'''
from sparsity import sparsity

sparse_mode = 'pattern_pruning'

if sparse_mode == 'sparse_pruning':
    sparsity = args.sparsity
    print(f'sparse_pruning {sparsity}')

elif sparse_mode == 'pattern_pruning':
    print(args.pattern_para)
    pattern_num   = int(args.pattern_para.split('_')[0])
    pattern_shape = [int(args.pattern_para.split('_')[1]), int(args.pattern_para.split('_')[2])]
    pattern_nnz   = int(args.pattern_para.split('_')[3])
    print(f'pattern_pruning {pattern_num} [{pattern_shape[0]}, {pattern_shape[1]}] {pattern_nnz}')
    patterns = generate_pattern(pattern_num, pattern_shape, pattern_nnz)
    pattern_mask = generate_pattern_mask(model, patterns)

elif sparse_mode == 'coo_pruning':
    coo_shape   = [int(args.coo_para.split('_')[0]), int(args.coo_para.split('_')[1])]
    coo_nnz   = int(args.coo_para.split('_')[2])
    # patterns = generate_pattern(pattern_num, pattern_shape, pattern_nnz)
    print(f'coo_pruning [{coo_shape[0]}, {coo_shape[1]}] {coo_nnz}')

elif sparse_mode == 'ptcoo_pruning':
    pattern_num   = int(args.pattern_para.split('_')[0])
    pattern_shape = [int(args.ptcoo_para.split('_')[1]), int(args.ptcoo_para.split('_')[2])]
    pt_nnz   = int(args.ptcoo_para.split('_')[3])
    coo_nnz   = int(args.ptcoo_para.split('_')[4])
    patterns = generate_pattern(pattern_num, pattern_shape, pt_nnz)
    pattern_mask = generate_pattern_mask(model, patterns)
    print(f'ptcoo_pruning {pattern_num} [{pattern_shape[0]}, {pattern_shape[1]}] {pt_nnz} {coo_nnz}')
