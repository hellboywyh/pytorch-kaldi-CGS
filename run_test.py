##########################################################
# pytorch-kaldi v.0.1
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# October 2018
##########################################################


from __future__ import print_function
from __future__ import division

import os
import sys
import glob
import configparser
# import ConfigParser as configparser
import numpy as np
import torch
from utils import check_cfg, create_lists, create_configs, compute_avg_performance, \
    read_args_command_line, run_shell, compute_n_chunks, get_all_archs, cfg_item2sec, \
    dump_epoch_results, create_curves, change_lr_cfg, expand_str_ep, model_init, optimizer_init
from shutil import copyfile
import re
from distutils.util import strtobool
import importlib
import math
import threading
from data_io import read_lab_fea, open_or_fd, write_mat
from pattern_search import pattern_certain_nnz_prun_model

# Reading global cfg file (first argument-mandatory file)
cfg_file = sys.argv[1]
if not (os.path.exists(cfg_file)):
    sys.stderr.write(
        'ERROR: The config file %s does not exist!\n' % (cfg_file))
    sys.exit(0)
else:
    config = configparser.ConfigParser()
    config.read(cfg_file)

# Reading and parsing optional arguments from command line (e.g.,--optimization,lr=0.002)
[section_args, field_args, value_args] = read_args_command_line(
    sys.argv, config)

# Output folder creation
out_folder = config['exp']['out_folder']
if not os.path.exists(out_folder):
    os.makedirs(out_folder + '/exp_files')

# Log file path
log_file = config['exp']['out_folder'] + '/log.log'

# Read, parse, and check the config file
cfg_file_proto = config['cfg_proto']['cfg_proto']
[config, name_data, name_arch] = check_cfg(cfg_file, config, cfg_file_proto)

# Read cfg file options
is_production = strtobool(config['exp']['production'])
cfg_file_proto_chunk = config['cfg_proto']['cfg_proto_chunk']

cmd = config['exp']['cmd']
N_ep = int(config['exp']['N_epochs_tr'])
N_ep_str_format = '0' + str(int(max(math.ceil(np.log10(N_ep)), 1))) + 'd'
tr_data_lst = config['data_use']['train_with'].split(',')
valid_data_lst = config['data_use']['valid_with'].split(',')
forward_data_lst = config['data_use']['forward_with'].split(',')
max_seq_length_train = config['batches']['max_seq_length_train']
forward_save_files = list(
    map(strtobool, config['forward']['save_out_file'].split(',')))

if config.has_option('exp', 'apply_prune_ep'):
    apply_prune_ep = int(config.get('exp', 'apply_prune_ep'))
else:
    apply_prune_ep = 0

print("- Reading config file......OK!")

# Copy the global cfg file into the output folder
cfg_file = out_folder + '/conf.cfg'
with open(cfg_file, 'w') as configfile:
    config.write(configfile)

# Load the run_nn function from core libriary
# The run_nn is a function that process a single chunk of data
run_nn_script = config['exp']['run_nn_script'].split('.py')[0]
module = importlib.import_module('core')
run_nn = getattr(module, run_nn_script)

# Splitting data into chunks (see out_folder/additional_files)
create_lists(config) 

# Writing the config files
create_configs(config)

print("- Chunk creation......OK!\n")

# create res_file
res_file_path = out_folder + '/res.res'
# res_file = open(res_file_path, "w")
# res_file.close()

# Learning rates and architecture-specific optimization parameters
arch_lst = get_all_archs(config)
lr = {}
auto_lr_annealing = {}
improvement_threshold = {}
halving_factor = {}
pt_files = {}

for arch in arch_lst:
    lr[arch] = expand_str_ep(config[arch]['arch_lr'], 'float', N_ep, '|', '*')
    if len(config[arch]['arch_lr'].split('|')) > 1:
        auto_lr_annealing[arch] = False
    else:
        auto_lr_annealing[arch] = True
    improvement_threshold[arch] = float(
        config[arch]['arch_improvement_threshold'])
    halving_factor[arch] = float(config[arch]['arch_halving_factor'])
    pt_files[arch] = config[arch]['arch_pretrain_file']

# If production, skip training and forward directly from last saved models
if is_production:
    ep = N_ep - 1
    N_ep = 0
    model_files = {}

    for arch in pt_files.keys():
        model_files[arch] = out_folder + '/exp_files/final_' + arch + '.pkl'

op_counter = 1  # used to dected the next configuration file from the list_chunks.txt

# Reading the ordered list of config file to process
cfg_file_list = [line.rstrip('\n') for line in open(
    out_folder + '/exp_files/list_chunks.txt')]
cfg_file_list.append(cfg_file_list[-1])

# A variable that tells if the current chunk is the first one that is being processed:
processed_first = True

data_name = []
data_set = []
data_end_index = []
fea_dict = []
lab_dict = []
arch_dict = []

# --------Pruning model and save pruned model--------#

# Reading all the features and labels for this chunk
ep = N_ep - 1
ck = 0
N_ck_forward = compute_n_chunks(
    out_folder, forward_data_lst[0], ep, N_ep_str_format, 'forward')
N_ck_str_format = '0' + \
    str(int(max(math.ceil(np.log10(N_ck_forward)), 1))) + 'd'
config_chunk_file = out_folder + '/exp_files/forward_' + forward_data_lst[0] + '_ep' + format(ep, N_ep_str_format)\
    + '_ck' + format(ck, N_ck_str_format) + '.cfg'
shared_list = []
test_config = configparser.ConfigParser()
test_cfg_file = config_chunk_file
test_config.read(test_cfg_file)
output_folder = test_config['exp']['out_folder']
is_production = strtobool(test_config['exp']['production'])
model = test_config['model']['model'].split('\n')
use_cuda = strtobool(test_config['exp']['use_cuda'])
multi_gpu = strtobool(test_config['exp']['multi_gpu'])
to_do = test_config['exp']['to_do']
info_file = test_config['exp']['out_info']

p = threading.Thread(target=read_lab_fea, args=(
    test_cfg_file, is_production, shared_list, output_folder,))
p.start()
p.join()

data_name = shared_list[0]
data_end_index = shared_list[1]
fea_dict = shared_list[2]
lab_dict = shared_list[3]
arch_dict = shared_list[4]
data_set = shared_list[5]

[nns, costs] = model_init(fea_dict, model, test_config,
                          arch_dict, use_cuda, multi_gpu, to_do)
# optimizers initialization
optimizers = optimizer_init(nns, test_config, arch_dict)

# load pre-training model
for net in nns.keys():
    pt_file_arch = config[arch_dict[net][0]]['arch_pretrain_file']
    print(pt_file_arch)
    if pt_file_arch != 'none':
        checkpoint_load = torch.load(pt_file_arch)
        nns[net].load_state_dict(checkpoint_load['model_par'])
        optimizers[net].load_state_dict(checkpoint_load['optimizer_par'])
#         optimizers[net].param_groups[0]['lr'] = float(
#             test_config[arch_dict[net][0]]['arch_lr'])  # loading lr of the cfg file for pt

# pattern prun model
if_pattern_prun = strtobool(config['pattern']['pattern_prun'])
pattern_num = int(config['pattern']['pattern_num'])
pattern_shape = list(map(int, config['pattern']['pattern_shape'].split(',')))
pattern_nnz = int(config['pattern']['pattern_nnz'])

if if_pattern_prun:
    nns = pattern_certain_nnz_prun_model(
        nns, pattern_num, pattern_shape, pattern_nnz, if_pattern_prun=True)
    # save pattern pruned model
    for net in nns.keys():
        checkpoint = {}
        checkpoint['model_par'] = nns[net].state_dict()
        checkpoint['optimizer_par'] = optimizers[net].state_dict()

        out_file = info_file.replace(
            '.info', f'_{arch_dict[net][0]}_{pattern_num}_{pattern_shape[0]}x{pattern_shape[1]}_{pattern_nnz}_pattern.pkl')
#         out_file = test_config[arch_dict[net][0]]['arch_pretrain_file']
        print(out_file)
        torch.save(checkpoint, out_file)

# # modify pre_trained model in test cfg
# for ck in range(N_ck_forward):
#     config_chunk_file = out_folder + '/exp_files/forward_' + \
#         forward_data_lst[0] + '_ep' + \
#         format(ep, N_ep_str_format) + '_ck' + \
#         format(ck, N_ck_str_format) + '.cfg'
#     config_chunk = configparser.ConfigParser()
#     config_chunk.read(config_chunk_file)
#     for arch in arch_lst:
#         config_chunk[arch]["arch_pretrain_file"] = info_file.replace(
#             '.info', f'_{arch}_{pattern_num}_{pattern_shape[0]}x{pattern_shape[1]}_{pattern_nnz}_pattern.pkl')
#     # Write cfg_file_chunk
#     with open(config_chunk_file, 'w') as configfile:
#         config_chunk.write(configfile)


# --------FORWARD--------#
ep = N_ep - 1
for forward_data in forward_data_lst:

    # Compute the number of chunks
    N_ck_forward = compute_n_chunks(
        out_folder, forward_data, ep, N_ep_str_format, 'forward')
    N_ck_str_format = '0' + \
        str(int(max(math.ceil(np.log10(N_ck_forward)), 1))) + 'd'
    print("N_ck_forward:",N_ck_forward)
    for ck in range(N_ck_forward):

        if not is_production:
            print('Testing %s chunk = %i / %i' %
                  (forward_data, ck + 1, N_ck_forward))
        else:
            print('Forwarding %s chunk = %i / %i' %
                  (forward_data, ck + 1, N_ck_forward))

        # output file
        info_file = out_folder + '/exp_files/forward_' + forward_data + '_ep' + format(ep,
                                                                                       N_ep_str_format) + '_ck' + format(
            ck, N_ck_str_format) + '.info'
        config_chunk_file = out_folder + '/exp_files/forward_' + forward_data + '_ep' + format(ep,
                                                                                               N_ep_str_format) + '_ck' + format(
            ck, N_ck_str_format) + '.cfg'

        # Do forward if the chunk was not already processed
        if not (os.path.exists(info_file)):
        # if True:
            # Doing forward

            # getting the next chunk
            next_config_file = cfg_file_list[op_counter]

            # run chunk processing
            [data_name, data_set, data_end_index,
             fea_dict, lab_dict, arch_dict] = run_nn(data_name, data_set,
                                                     data_end_index, fea_dict,
                                                     lab_dict, arch_dict,
                                                     config_chunk_file,
                                                     processed_first,
                                                     next_config_file,
                                                     if_pattern_search=False)

            # update the first_processed variable
            processed_first = False

            if not (os.path.exists(info_file)):
                sys.stderr.write(
                    "ERROR: forward chunk %i of dataset %s not done! File %s does not exist.\nSee %s \n" % (
                        ck, forward_data, info_file, log_file))
                sys.exit(0)

        # update the operation counter
        op_counter += 1

# --------DECODING--------#
dec_lst = glob.glob(out_folder + '/exp_files/*_to_decode.ark')

forward_data_lst = config['data_use']['forward_with'].split(',')
forward_outs = config['forward']['forward_out'].split(',')
forward_dec_outs = list(
    map(strtobool, config['forward']['require_decoding'].split(',')))

for data in forward_data_lst:
    for k in range(len(forward_outs)):
        if forward_dec_outs[k]:

            print('Decoding %s output %s' % (data, forward_outs[k]))

            info_file = out_folder + '/exp_files/decoding_' + \
                data + '_' + forward_outs[k] + '.info'

            # create decode config file
            config_dec_file = out_folder + '/decoding_' + \
                data + '_' + forward_outs[k] + '.conf'
            config_dec = configparser.ConfigParser()
            config_dec.add_section('decoding')

            for dec_key in config['decoding'].keys():
                config_dec.set('decoding', dec_key,
                               config['decoding'][dec_key])

            # add graph_dir, datadir, alidir
            lab_field = config[cfg_item2sec(config, 'data_name', data)]['lab']

            # Production case, we don't have labels
            if not is_production:
                pattern = 'lab_folder=(.*)\nlab_opts=(.*)\nlab_count_file=(.*)\nlab_data_folder=(.*)\nlab_graph=(.*)'
                alidir = re.findall(pattern, lab_field)[0][0]
                config_dec.set('decoding', 'alidir', os.path.abspath(alidir))

                datadir = re.findall(pattern, lab_field)[0][3]
                config_dec.set('decoding', 'data', os.path.abspath(datadir))

                graphdir = re.findall(pattern, lab_field)[0][4]
                config_dec.set('decoding', 'graphdir',
                               os.path.abspath(graphdir))
            else:
                pattern = 'lab_data_folder=(.*)\nlab_graph=(.*)'
                datadir = re.findall(pattern, lab_field)[0][0]
                config_dec.set('decoding', 'data', os.path.abspath(datadir))

                graphdir = re.findall(pattern, lab_field)[0][1]
                config_dec.set('decoding', 'graphdir',
                               os.path.abspath(graphdir))

                # The ali dir is supposed to be in exp/model/ which is one level ahead of graphdir
                alidir = graphdir.split('/')[0:len(graphdir.split('/')) - 1]
                alidir = "/".join(alidir)
                config_dec.set('decoding', 'alidir', os.path.abspath(alidir))

            with open(config_dec_file, 'w') as configfile:
                config_dec.write(configfile)

            out_folder = os.path.abspath(out_folder)
            files_dec = out_folder + '/exp_files/forward_' + data + \
                '_ep*_ck*_' + forward_outs[k] + '_to_decode.ark'
            out_dec_folder = out_folder + '/decode_' + \
                data + '_' + forward_outs[k]

            if not (os.path.exists(info_file)):

                # Run the decoder
                cmd_decode = cmd + config['decoding']['decoding_script_folder'] + '/' + config['decoding'][
                    'decoding_script'] + ' ' + os.path.abspath(
                    config_dec_file) + ' ' + out_dec_folder + ' \"' + files_dec + '\"'
                run_shell(cmd_decode, log_file)

                # # remove ark files if needed
                # if not forward_save_files[k]:
                #     list_rem = glob.glob(files_dec)
                #     for rem_ark in list_rem:
                #         os.remove(rem_ark)

            # Print WER results and write info file
            cmd_res = './check_res_dec.sh ' + out_dec_folder
            wers = run_shell(cmd_res, log_file).decode('utf-8')
            res_file = open(res_file_path, "a")
            res_file.write('%s\n' % wers)
            print(wers)
