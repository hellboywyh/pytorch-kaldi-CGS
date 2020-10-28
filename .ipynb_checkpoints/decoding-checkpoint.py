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
from pattern_search import pattern_prun_model

cfg_file = '/yhwang/0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/cfg/20201021_Pattern_Search/TIMIT_LSTM_fmllr_L2_8bw_16ba_wohcgs_v1.cfg'

if not (os.path.exists(cfg_file)):
    sys.stderr.write(
        'ERROR: The config file %s does not exist!\n' % (cfg_file))
    sys.exit(0)
else:
    config = configparser.ConfigParser()
    config.read(cfg_file)

    # Log file path
log_file = config['exp']['out_folder'] + '/log.log'

# Read, parse, and check the config file
cfg_file_proto = config['cfg_proto']['cfg_proto']
[config, name_data, name_arch] = check_cfg(cfg_file, config, cfg_file_proto)

out_folder = config['exp']['out_folder']
forward_save_files = list(
    map(strtobool, config['forward']['save_out_file'].split(',')))
is_production = strtobool(config['exp']['production'])
cmd = config['exp']['cmd']
res_file_path = out_folder + '/res.res'

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
            print(info_file)
            if not (os.path.exists(info_file)):

                # Run the decoder
                cmd_decode = cmd + config['decoding']['decoding_script_folder'] + '/' + config['decoding'][
                    'decoding_script'] + ' ' + os.path.abspath(
                    config_dec_file) + ' ' + out_dec_folder + ' \"' + files_dec + '\"'
                print(cmd_decode, log_file)
                run_shell(cmd_decode, log_file)

                # remove ark files if needed
                if not forward_save_files[k]:
                    list_rem = glob.glob(files_dec)
                    for rem_ark in list_rem:
                        os.remove(rem_ark)

            # Print WER results and write info file
            cmd_res = './check_res_dec.sh ' + out_dec_folder
            print(cmd_res, log_file)
            wers = run_shell(cmd_res, log_file).decode('utf-8')
            res_file = open(res_file_path, "a")
            res_file.write('%s\n' % wers)
            print(wers)
