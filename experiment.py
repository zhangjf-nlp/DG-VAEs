#!/usr/bin/env python
# coding: utf-8

from local_info import machine_name
from cuda_utils import dynamic_cuda_allocation
dynamic_cuda_allocation()

import os
import sys
import time
import numpy as np
from collections import defaultdict

import importlib
import argparse

import torch
from torch import nn, optim

from data import MonoTextData, VocabEntry
from modules.vae import VAE
from modules.decoders.decoder import LSTMDecoder
from modules.encoders.encoder import available_encoder_classes
from utils import uniform_initializer, calc_mi, calc_au

clip_grad = 5.0
decay_epoch = 5
lr_decay = 0.5
max_decay = 5

def init_config(args_specification = None):
    # for how to specify the arguments for different datasets and different methods
    # please refer to **args_settings.py**
    
    parser = argparse.ArgumentParser(description='For systematic experiments on VAEs')
    
    # model hyperparameters
    parser.add_argument('--dataset', type=str, required=True, help='dataset to use')
    
    # optimization parameters
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0, help='sgd momentum')
    parser.add_argument('--opt', type=str, choices=["sgd", "adam"], default="sgd", help='sgd momentum')
    parser.add_argument("--reset_dec", action="store_true", default=False)
    parser.add_argument("--load_best_epoch", type=int, default=15)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument("--verbose", type=int, default=0)
    
    # annealing paramters
    parser.add_argument('--warm_up', type=int, default=10, help="number of annealing epochs. warm_up=0 means not anneal")
    parser.add_argument('--kl_start', type=float, default=0.0, help="starting KL weight")
    
    # inference parameters
    parser.add_argument('--seed', type=int, default=783435, metavar='S', help='random seed')
    
    # output directory
    parser.add_argument('--exp_dir', default=None, type=str, help='experiment directory.')
    
    # method-specific parameters
    parser.add_argument("--dim_z", type=int, default=0, help="specify the latent dimension")
    parser.add_argument("--encoder_class", type=str, choices=list(available_encoder_classes.keys()), help="the name of implemented encoder class", default="GaussianLSTMEncoder")
    parser.add_argument('--cycle', type=int, default=0, help="for cyclic-vae")
    parser.add_argument("--add_skip", action="store_true", default=False, help="for skip-VAE")
    parser.add_argument("--add_bow", action="store_true", default=False, help="for bow-VAE")
    parser.add_argument("--gamma", type=float, default=0.7, help="for bn-vae")
    parser.add_argument("--target_kl", type=float, default=4.0, help="for FB-VAE")
    parser.add_argument("--delta", type=float, default=0.15, help="for Delta-VAE")
    parser.add_argument("--kappa", type=int, default=13, help="for vMF-VAE")
    parser.add_argument("--kl_beta", type=float, default=1.0, help="for Beta-VAE")
    parser.add_argument("--agg_size", type=int, default=None, help="for ablation study on DG-VAE -> VAE")
    
    if args_specification:
        args = parser.parse_args([_ for _ in args_specification if _])
    else:
        args = parser.parse_args()
    
    # set args.cuda
    args.cuda = torch.cuda.is_available()
    
    # set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # load config file into args
    config_file = "config.config_%s" % args.dataset
    config_module = importlib.import_module(config_file)
    params = config_module.params
    args = argparse.Namespace(**vars(args), **params)
    args.dim_z = args.dim_z if args.dim_z else args.nz
    args.nz = args.dim_z
    args.embedding_size = args.ni
    args.hidden_size = args.enc_nh
    
    prefix = {
        "GaussianLSTMEncoder":"", # VAE (default)
        "BNGaussianLSTMEncoder":"BN-", # BN-VAE
        "DeltaGaussianLSTMEncoder":"Delta-", # Delta-VAE
        "FineFBGaussianLSTMEncoder":"FB-", # FB-VAE
        "CoarseFBGaussianLSTMEncoder":"CFB-", # another kind of FB-VAE, not reported in paper
        "DGGaussianLSTMEncoder":"DG-", # proposed DG-VAE
        "VMFLSTMEncoder":"vMF-", # vMF-VAE
        "DGVMFLSTMEncoder":"DG-vMF-", # proposed DG-vMF-VAE
    }[args.encoder_class]
    postfix = {
        "GaussianLSTMEncoder":"", # VAE (default)
        "BNGaussianLSTMEncoder":"({})".format(args.gamma), # BN-VAE
        "DeltaGaussianLSTMEncoder":"({})".format(args.delta), # Delta-VAE
        "FineFBGaussianLSTMEncoder":"({})".format(args.target_kl), # FB-VAE
        "CoarseFBGaussianLSTMEncoder":"({})".format(args.target_kl), # another kind of FB-VAE, not reported in paper
        "DGGaussianLSTMEncoder":"({})".format(args.agg_size) if args.agg_size else "", # proposed DG-VAE
        "VMFLSTMEncoder":"({})".format(args.kappa), # vMF-VAE
        "DGVMFLSTMEncoder":"({})".format(args.kappa), # proposed DG-vMF-VAE
    }[args.encoder_class]
    
    if args.add_skip:
        prefix = "skip-" + prefix
    if args.add_bow:
        prefix = "bow-" + prefix
    if args.cycle > 0:
        prefix = "cycle-" + prefix
    if args.kl_beta != 1.0:
        prefix = "beta-" + prefix
        postfix += "({})".format(args.kl_beta)
    if args.dim_z != 32:
        postfix += "-dim_z({})".format(args.dim_z)
    
    args.model_name = f"{prefix}VAE{postfix}"
    
    # set load and save paths
    if args.exp_dir == None:
        args.exp_dir = "exp_{}/{}".format(args.dataset, args.model_name)
    
    args.save_path = os.path.join(args.exp_dir, 'model.pt')
    
    # set args.label
    if 'label' in params:
        args.label = params['label']
    else:
        args.label = False
    return args

@torch.no_grad()
def test(model, test_data_batch, mode, args, logging, verbose=True):
    metric_names = ["loss_eval", "elbo", "kl", "okl", "logp_prior", "logp_post"]
    total_values = {metric_name:0.0 for metric_name in metric_names}
    num_samples = 0
    for i in range(len(test_data_batch)):
        batch_data = test_data_batch[i]
        batch_size, seq_len = batch_data.size()
        metric_values = model.evaluate(batch_data)
        for metric_name, metric_value in zip(metric_names, metric_values):
            total_values[metric_name] += metric_value*batch_size
        num_samples += batch_size
    mean_values = {metric_name:total_values[metric_name]/num_samples
                   for metric_name in metric_names}
    mean_values["mi"] = calc_mi(model, test_data_batch)
    mean_values["au"], au_var = calc_au(model, test_data_batch)
    metric_names += ["mi", "au"]
    if verbose:
        logging(f'{mode} --- {", ".join([f"{metric_name}: {mean_values[metric_name]:.4f}" for metric_name in metric_names])}')
    return mean_values["loss_eval"], mean_values

@torch.no_grad()
def test_classification(model, test_data_batch, test_labels_batch, mode, args, logging, verbose=True):
    metric_names = ["loss_eval", "acc"]
    total_values = {metric_name:0.0 for metric_name in metric_names}
    num_samples = 0
    for i in range(len(test_data_batch)):
        batch_data = test_data_batch[i]
        batch_labels = torch.tensor([int(x) for x in test_labels_batch[i]]).long().cuda()
        batch_size, seq_len = batch_data.size()
        metric_values = model.evaluate(batch_data, batch_labels)
        for metric_name, metric_value in zip(metric_names, metric_values):
            total_values[metric_name] += metric_value*batch_size
        num_samples += batch_size
    mean_values = {metric_name:total_values[metric_name]/num_samples
                   for metric_name in metric_names}
    if verbose:
        logging(f'{mode} --- {", ".join([f"{metric_name}: {mean_values[metric_name]:.4f}" for metric_name in metric_names])}')
    return mean_values["loss_eval"], mean_values

def create_model(args, vocab):
    encoder_class = available_encoder_classes[args.encoder_class]
    if args.verbose:
        print("creating vae ...")
    args.vocab_size = len(vocab)
    model_init = uniform_initializer(0.01)
    emb_init = uniform_initializer(0.1)
    encoder = encoder_class(args, model_init, emb_init)
    decoder = LSTMDecoder(args, vocab, model_init, emb_init)
    vae = VAE(encoder, decoder, args).to(args.device)
    if args.verbose:
        print(f"vae: {vae}")
    return vae

def create_exp_dir(dir_path, debug=False):
    import functools, shutil
    log_path = os.path.join(dir_path, 'log.txt')
    def logging(s, log_path, print_=True):
        if log_path is not None:
            with open(log_path, 'a+') as f_log:
                f_log.write(s + '\n')
        if print_:
            print(s)
    if debug:
        print('Debug Mode : no experiment dir created')
        return functools.partial(logging, log_path=None)
    if os.path.exists(dir_path):
        print("Path {} exists. Remove and remake.".format(dir_path))
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    print('Experiment dir : {}'.format(dir_path))
    return functools.partial(logging, log_path=log_path)

def create_optimizer(args, model, opt_dict):
    if args.opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=opt_dict["lr"], momentum=args.momentum)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=opt_dict["lr"])
    else:
        raise ValueError("optimizer not supported")
    return optimizer

def optimize_kl_over_epoch(args, model, data, kl_weight, logging=print, num_batch=3, n_samples=4):
    from modules.utils import exp_mean_log
    model.train()
    batch_size = 128
    data_batch = data.create_data_batch(batch_size=batch_size, device=args.device, batch_first=True)
    optimizer = optim.Adam(model.encoder.parameters(), lr=1e-4)
    list_mean, list_logvar = [], []
    for i in range(num_batch):
        for j in np.random.permutation(len(data_batch)):
            batch_data = data_batch[j]
            if batch_data.shape[0] < 2:
                continue
            mean, logvar = model.encoder.input_to_posterior(batch_data)
            list_mean.append(mean)
            list_logvar.append(logvar)
            if sum([_.shape[0] for _ in list_mean])>=batch_size*4:
                mean = torch.cat(list_mean, dim=0)
                logvar = torch.cat(list_logvar, dim=0)
                z = model.encoder.posterior_to_zs(mean, logvar, nsamples=n_samples).view(-1,args.dim_z)
                logpz = (-np.log(2*np.pi)/2 - z.square()/2).sum(dim=-1) # [n_data*n_samples]
                logqz_given_x = (-np.log(2*np.pi)/2 - logvar[:,None,:]/2 - ((z[None,:,:]-mean[:,None,:]).square()+1e-6) / (2*torch.exp(logvar[:,None,:])+1e-6)).sum(dim=-1) # [n_data, n_data*n_samples]
                logqz = exp_mean_log(logqz_given_x, dim=0)
                loss = (logqz - logpz).mean(dim=0) * kl_weight
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                optimizer.zero_grad()
                list_mean, list_logvar = [], []
                if not loss.item()==loss.item():
                    import pdb;pdb.set_trace()
        logging(f"KL(q(z)||p(z)) = {loss.item():.2f}")

def main(args, logging):
    opt_dict = {"not_improved": 0, "lr": 1., "best_loss": 1e4}
    
    args.device = "cuda" if args.cuda else "cpu"
    
    train_data = MonoTextData(args.train_data, label=args.label)
    vocab = train_data.vocab
    val_data = MonoTextData(args.val_data, label=args.label, vocab=vocab)
    test_data = MonoTextData(args.test_data, label=args.label, vocab=vocab)
    
    if args.verbose:
        logging('Train data: %d samples' % len(train_data))
        logging('finish reading datasets, vocab size is %d' % len(vocab))
        logging('dropped sentences: %d' % train_data.dropped)
    
    log_niter = (len(train_data)//args.batch_size)//10
    
    model = create_model(args, vocab)
    opt_dict['lr'] = args.lr if args.opt == "sgd" else 0.001
    optimizer = create_optimizer(args, model, opt_dict)
    
    iter_ = decay_cnt = 0
    model.train()
    start = time.time()
    
    kl_weight = args.kl_start # this should have no influence on training classifier
    if args.warm_up > 0:
        anneal_rate = (1.0 - args.kl_start) / (args.warm_up * (len(train_data) / args.batch_size))
    else:
        anneal_rate = 0
    
    train_data_batch, val_data_batch, test_data_batch = [
        data.create_data_batch(batch_size=args.batch_size, device=args.device, batch_first=True)
        for data in [train_data, val_data, test_data]]
    logging(f'train-data: {len(train_data_batch)}, eval-data: {len(val_data_batch)}, test-data: {len(test_data_batch)}')
    
    metric_names = ["num_samples", "loss_train", "loss_rec_train", "loss_kld_train", "kl_weight"]
    train_metric_values = {metric_name:0 for metric_name in metric_names}
    
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(args.epochs):
            if args.cycle > 0 and (epoch - 1) % args.cycle == 0:
                kl_weight = args.kl_start
                print('KL Annealing restart!')
                        
            model.train()
            for i in np.random.permutation(len(train_data_batch)):
                batch_data = train_data_batch[i]
                batch_size, sent_len = batch_data.size()
                if batch_size < 2:
                    continue
                #if args.agg_size and not batch_size%args.agg_size==0:
                #    continue
                train_metric_values["num_samples"] += batch_size
                kl_weight = min(1.0, kl_weight + anneal_rate)
                loss, loss_rec, loss_kld = model(batch_data, kl_weight)
                train_metric_values["loss_train"] += loss.sum().item()
                train_metric_values["loss_rec_train"] += loss_rec.sum().item()
                train_metric_values["loss_kld_train"] += loss_kld.item()*batch_size
                train_metric_values["kl_weight"] += kl_weight
                
                loss = loss.mean(dim=-1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                optimizer.zero_grad()
                
                if iter_ % log_niter == 0:
                    if args.verbose:
                        report_metrics = [f'{metric_name}: {train_metric_values[metric_name]/metric_name["num_samples"]:%.4f}'
                                          for metric_name in metric_names[1:]]
                        logging(f'epoch: {epoch}, iter: {iter_}, time: {time.time() - start:%.2fs}, {", ".join(report_metrics)}')
                    train_metric_values = {metric_name:0 for metric_name in metric_names}
                
                iter_ += 1
            
            if args.verbose:
                logging('kl weight %.4f' % kl_weight)
                logging('lr {}'.format(opt_dict["lr"]))
            
            model.eval()
            logging('evaluation at epoch: %d'%epoch)
            loss_eval, metrics = test(model, val_data_batch, "VAL", args, logging)
            
            if loss_eval > opt_dict["best_loss"]:
                opt_dict["not_improved"] += 1
                if opt_dict["not_improved"] >= decay_epoch and epoch >= args.load_best_epoch:
                    opt_dict["best_loss"] = loss
                    opt_dict["not_improved"] = 0
                    opt_dict["lr"] = opt_dict["lr"] * lr_decay
                    model.load_state_dict(torch.load(args.save_path))
                    if args.verbose:
                        logging('new lr: %f' % opt_dict["lr"])
                    decay_cnt += 1
                    optimizer = create_optimizer(args, model, opt_dict)
            else:
                if args.verbose:
                    logging(f'update best loss: {opt_dict["best_loss"]} -> {loss_eval}')
                opt_dict["not_improved"] = 0
                opt_dict["best_loss"] = loss_eval
                torch.save(model.state_dict(), args.save_path)
            
            if decay_cnt == max_decay and args.cycle == 0:
                logging('-' * 100)
                logging('Exiting from training for max_decay')
                break
            
            if epoch % args.test_nepoch == 0:
                loss_test, metrics = test(model, test_data_batch, "TEST", args, logging)
        
    except KeyboardInterrupt:
        logging('-' * 100)
        logging('Exiting from training for KeyboardInterrupt')
    except Exception as e:
        raise e
    
    model.load_state_dict(torch.load(args.save_path))
    model.eval()
    loss_test, metrics = test(model, test_data_batch, "FINAL-TEST", args, logging)

if __name__ ==  "__main__":
    args = init_config()
    logging = create_exp_dir(args.exp_dir, debug=args.eval)
    logging("This experiement started at: {} on: {}".format(time.ctime(), machine_name)) # historical type error ...
    logging(str(args))
    try:
        main(args, logging)
    except Exception as e:
        import traceback
        logging(f'traceback.format_exc():\n{traceback.format_exc()}')