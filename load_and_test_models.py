import os
import re
import sys
import time
import math
import datetime
import numpy as np
from collections import defaultdict

from cuda_utils import dynamic_cuda_allocation
dynamic_cuda_allocation()

import importlib
import argparse

import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from data import MonoTextData

from utils import uniform_initializer, calc_mi, calc_au
from utils import visualize2D_posterior_distribution

from modules.utils import log_sum_exp, exp_mean_log
from modules.encoders.utils import GaussianPDF, LogGaussianPDF

from experiment import init_config, create_model, test

from local_info import machine_name

metric_names_lm = ["loss_eval", "elbo", "kl", "okl", "logp_prior", "logp_post", "mi", "au"]
CU_thresh = [0.01, 0.03, 0.05, 0.1] # we report the results of 0.03 in paper for better comparison
metric_names_latent = ["joint-kl"] + [f"CU({thresh})" for thresh in CU_thresh]

def check_args(args):
    """
    jsut check wether the log.txt under such directories ended with "FINAL-TEST",
    so as to find those directories with their experiments interrupted.
    """
    log_path = args.save_path.replace("model.pt", "log.txt")
    try:
        with open(log_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        return False
    lines = content.split("\n")
    final_test_finished = any(["FINAL-TEST" in line for line in lines])
    return final_test_finished

def get_args_status(args):
    log_path = args.save_path.replace("model.pt", "log.txt")
    exist_flag = os.path.exists(log_path)
    create_time = time.mktime(time.strptime(
        re.match("This experiement started at: (.*) on: .*",
                 open(log_path,'r').readlines()[0]).group(1), '%c')) if exist_flag else None
    modify_time = max(os.stat(log_path).st_mtime, create_time) if exist_flag else None
    active_flag = exist_flag and (time.time() - modify_time < 600)
    complete_flag = check_args(args)
    return complete_flag, active_flag, exist_flag, log_path, create_time, modify_time

def display_args_status(args):
    complete_flag, active_flag, exist_flag, log_path, create_time, modify_time = get_args_status(args)
    print(f"{'complete' if complete_flag else 'incomplete'} {'-'*5 if complete_flag else '-'*3} "
        f"{log_path+' '*max(0,(48-len(log_path)))}" \
        f"\t{time.ctime(create_time) if exist_flag else '???'}" \
        f" -> {time.ctime(modify_time) if complete_flag else ('active' if active_flag else '???')}" \
        f" : {str(datetime.timedelta(seconds=int(modify_time-create_time))) if exist_flag else '--:--:--'}")
    return complete_flag, active_flag, exist_flag, log_path, create_time, modify_time

def clear_output():
    if 'ipykernel' in sys.modules:
        from IPython.display import clear_output as clear
        clear()
    else:
        os.system('cls' if os.name == 'nt' else 'clear')
    return

def block_until_all_complete(args_specifications, interval=60):
    args_list = [init_config(args_specification) for args_specification in args_specifications]
    try:
        while True:
            clear_output()
            print(f"{time.ctime()} - {machine_name}")
            complete_flags = [display_args_status(args)[0] for args in args_list]
            if all(complete_flags):
                break
            else:
                time.sleep(interval)
    except KeyboardInterrupt:
        print("stop waiting for these experiments to complete")
        return False
    return True

def test_preparation(args_specification, args=None):
    """ preparation for testing a model:
    1. args
    2. model
    3. data + vocab
    """
    if args is None:
        args = init_config(args_specification)
    args.device = "cuda" if args.cuda else "cpu"
    
    train_data = MonoTextData(args.train_data, label=args.label)
    vocab = train_data.vocab
    val_data = MonoTextData(args.val_data, label=args.label, vocab=vocab)
    test_data = MonoTextData(args.test_data, label=args.label, vocab=vocab)
    
    model = create_model(args, vocab)
    model.load_state_dict(torch.load(args.save_path))
    model.eval()
    
    return args, model, (train_data, val_data, test_data), vocab

@torch.no_grad()
def get_encodings(args_specification, tp=None, encoded_data="test", verbose=0):
    args, model, (train_data, val_data, test_data), vocab = test_preparation(args_specification) if tp is None else tp
    if not model.encoder.useGaussian and False: # why not ?
        return args, None, None, None, None
    data = {"train": train_data, "val": val_data, "test": test_data}[encoded_data]
    data_batch = data.create_data_batch(
        batch_size=args.batch_size, device=args.device, batch_first=True)
    list_mean, list_logvar, list_textids = [], [], []
    bar = tqdm(data_batch) if verbose else data_batch
    for batch_data in bar:
        batch_size, seq_len = batch_data.shape
        mean, logvar = model.encoder.input_to_posterior(batch_data)
        list_mean.append(mean.cpu().numpy())
        list_logvar.append(logvar.cpu().numpy())
        list_textids += batch_data.tolist()
    all_mean = np.concatenate(list_mean, axis=0)
    all_logvar = np.concatenate(list_logvar, axis=0)
    all_textids = list_textids
    return args, all_mean, all_logvar, all_textids, vocab

def visualize_args_latent_space(args_specification, pltm, select_axis, mode, mask_area=None, sort_key=None, verbose=1, model_name=None, dataset=None):
    encodings = get_encodings(args_specification)
    args, all_mean, all_logvar, all_textids, vocab = encodings
    if any([_ is None for _ in [all_mean, all_logvar, all_textids, vocab]]):
        print(f"Not a gaussian model: {args.model_name}")
        return
    for sa,md in [(sa,md) for sa in select_axis for md in mode]:
        ax = pltm.subplot()
        visualize2D_posterior_distribution(all_mean, all_logvar, pltm.plt, ax, md, sa, mask_area)
        if verbose:
            pltm.plt.text(0, 1, f"  {args.model_name}", size = 10,
                     family = "fantasy", color = "black", style = "italic", weight = "light",
                     ma = 'left', ha = 'left', va = "bottom", transform=ax.transAxes)
            pltm.plt.text(0, 1, f"  {args.dataset}", size = 10, rotation=-90,
                     family = "fantasy", color = "black", style = "italic", weight = "light",
                     ma = 'left', ha = 'right', va = "top", transform=ax.transAxes)
        elif dataset is None:
            pltm.plt.title(model_name if model_name else args.model_name, fontsize=18)
        else:
            pltm.plt.title(f"{dataset} - {model_name}", fontsize=15)
    torch.cuda.empty_cache()
    if sort_key is not None and len(select_axis)==1 and type(select_axis[0]) is tuple:
        # mask an area in specific dimensions and output the corresponding sentences in specific order
        # empirically, it demonstrates the phenomenon that
        # the most active unit in vanilla vae is highly relevant to the length of sentence
        if mask_area is None:
            mask_area = lambda xy:True # print all samples
        num_all_text = all_mean.shape[0]
        masked_samples = []
        for i in range(all_mean.shape[0]):
            mu_x, mu_y = all_mean[i,select_axis[0][0]], all_mean[i,select_axis[0][1]]
            if not mask_area((mu_x, mu_y)):
                continue
            masked_samples.append((i, mu_x, mu_y))
        masked_samples = sorted(masked_samples, key=lambda ixy:sort_key((ixy[1],ixy[2])))
        masked_texts = vocab.decode_sentences([all_textids[i] for i,_,_ in masked_samples])
    else:
        masked_texts = None
    return masked_texts

@torch.no_grad()
def test_kl_hole(args_specification, tp=None, encodings=None, times_sampling=1000, samples_per_sampling=128, rt_unit_info=False):
    """ for gaussian latent models,  estimate:
        1. joint_kl: E_{z \sim p(z)} [logp(z)-logq(z)]
        2. marginal_kl: E_{z_i \sim p(z_i)} [logp(z_i)-logq(z_i)] for i in range(dim_z)
        3. covariance: Var_{x \sim D} [q(z_i|x).mean]
    if rt_unit_info is True, return unit-wised marginal_kl and covariance
    else (default), return joint_kl and CU
    """
    if encodings is None:
        encodings = get_encodings(args_specification, tp=tp)
    args, all_mean, all_logvar, all_textids, vocab = encodings
    assert "Gaussian" in args.encoder_class, "this method only supports Gaussian latent models"
    all_mean, all_logvar = torch.tensor(all_mean).to(args.device), torch.tensor(all_logvar).to(args.device)
    E_joint_kl, E_marginal_kl = [], []
    with torch.no_grad():
        for i in range(times_sampling):
            z = torch.zeros(samples_per_sampling, args.dim_z).normal_().to(args.device)
            # samples from prior for MC estimation: [samples_per_sampling, dim_z]
            
            m_log_q_z = LogGaussianPDF(
                all_mean, # [num_test_data, dim_z]
                all_logvar, # [num_test_data, dim_z]
                z.unsqueeze(0).repeat(all_mean.shape[0],1,1), # [num_test_data, samples_per_sampling, dim_z]
            )
            # the marginal log density in R: [num_test_data, samples_per_sampling, dim_z]
            j_log_q_z = m_log_q_z.sum(dim=-1)
            # the joint log density in R^dim_z: [num_test_data, samples_per_sampling]
            
            m_log_aggq_z = exp_mean_log(m_log_q_z, dim=0) # [samples_per_sampling, dim_z]
            j_log_aggq_z = exp_mean_log(j_log_q_z, dim=0) # [samples_per_sampling]
            # averaged density along aggregation / across test datapoints 
            
            m_log_p_z = LogGaussianPDF(
                torch.zeros_like(z), # [samples_per_sampling, dim_z]
                torch.zeros_like(z), # [samples_per_sampling, dim_z]
                z.unsqueeze(1) # [samples_per_sampling, 1, dim_z]
            ).squeeze(1)
            # the marginal log density in R: [samples_per_sampling, dim_z]
            j_log_p_z = m_log_p_z.sum(dim=-1)
            # the joint log density in R^dim_z: [samples_per_sampling]
            
            m_kl = m_log_p_z - m_log_aggq_z # [samples_per_sampling, dim_z]
            j_kl = j_log_p_z - j_log_aggq_z # [samples_per_sampling]
            E_marginal_kl.append(m_kl.mean(dim=0).tolist())
            E_joint_kl.append(j_kl.mean(dim=0).tolist())
    joint_kl = np.mean(E_joint_kl)
    marginal_kl = np.mean(E_marginal_kl, axis=0).tolist()
    variance_mean = torch.var(all_mean, dim=0).tolist()
    if rt_unit_info:
        return marginal_kl, variance_mean
    continuous_units = [sum([i<thresh for i in marginal_kl]) for thresh in CU_thresh]
    return {metric_name:metric_value for metric_name,metric_value in zip(metric_names_latent,[joint_kl]+continuous_units)}

@torch.no_grad()
def test_generation(args_specification, tp=None, num_intervals=10, mode="interpolation", erase=False, verbose=0):
    tp = test_preparation(args_specification) if tp is None else tp
    args, model, (train_data, val_data, test_data), vocab = tp
    test_data = val_data if mode in ["devlpsampling"] else test_data
    test_data_batch = test_data.create_data_batch(
        batch_size=args.batch_size, device=args.device, batch_first=True)
    if not model.encoder.useGaussian:
        return
    file_name, end_line = {
        "interpolation":(f"{args.exp_dir}/interpolation.txt", "Interpolation Finished"),
        "priorsampling":(f"{args.exp_dir}/priorsampling.txt", "Priorsampling Finished"),
    }[mode]
    if os.path.exists(file_name):
        existing_lines = open(file_name).readlines()
        if len(existing_lines)>0 and end_line in existing_lines[-1] and not erase:
            return # already finished -> do nothing
        with open(file_name, "w") as f:
            f.write("") # incomplete -> erase and test again
    content_output = ""
    bar = tqdm(range(len(test_data_batch))) if verbose else range(len(test_data_batch))
    for i in bar:
        batch_data = test_data_batch[i]
        batch_size, sent_len = batch_data.size()
        if batch_size < 2:
            continue
        mean, logvar, z, loss_kld = model.encoder(batch_data)
        if mode == "interpolation":
            for j in range(int(batch_size/2)):
                sample_a, sample_b = batch_data[j*2:j*2+1], batch_data[j*2+1:j*2+2]
                # [1, dim_z]
                mean_a, logvar_a = mean[j*2:j*2+1,:], logvar[j*2:j*2+1,:]
                mean_b, logvar_b = mean[j*2+1:j*2+2,:], logvar[j*2+1:j*2+2,:]
                # [num_intervals+1, 1]
                pb = torch.tensor(np.arange(num_intervals+1)/num_intervals).float().unsqueeze(-1).cuda()
                pa = 1-pb
                # [num_intervals+1, dim_z]
                means_a_to_b = torch.mm(pa,mean_a) + torch.mm(pb,mean_b)
                logvar_a_to_b = torch.mm(pa,logvar_a) + torch.mm(pb,logvar_b)
                # [num_intervals+1, dim_z]
                reconstruction = model.decode(means_a_to_b, strategy="greedy", K=5)
                content_output += "\n" + "a: "+" ".join([model.decoder.vocab.id2word(id) for id in sample_a.view(-1).tolist()])+"\n"
                content_output += "\n".join([f"{pa_:.2f}: {' '.join(words)}" for pa_, words in zip(pa.view(-1).tolist(), reconstruction)])
                content_output += "\n" + "b: "+" ".join([model.decoder.vocab.id2word(id) for id in sample_b.view(-1).tolist()])+"\n"
        elif mode == "priorsampling":
            zs = model.encoder.prior_to_zs(torch.zeros_like(mean), nsamples=1).squeeze(1)
            reconstruction = model.decode(zs, strategy="greedy", K=5)
            content_output += "\n" + "gts:" + "\n"
            content_output += "\n".join([" ".join([model.decoder.vocab.id2word(id) for id in batch_data[j].tolist()]) for j in range(batch_size)])
            content_output += "\n" + "res:" + "\n"
            content_output += "\n".join([" ".join(words) for words in reconstruction])
            
    with open(file_name, "a+") as f:
        f.write(content_output+"\n"+end_line)
    return

def test_args(args):
    tp = test_preparation(args_specification=None, args=args)
    args, model, (train_data, val_data, test_data), vocab = tp
    with torch.no_grad():
        # metrics for language modeling
        test_data_batch = test_data.create_data_batch(
            batch_size=args.batch_size, device=args.device, batch_first=True)
        loss_eval, metrics = test(model, test_data_batch, "TEST", args, print, verbose=0)
        # metrics for latent space
        if model.encoder.useGaussian:
            metrics.update(test_kl_hole(args_specification=None, tp=tp))
        else:
            metrics.update({metric_name:np.nan for metric_name in metric_names_latent}) # TODO
    return metrics, tp

def get_all_results(args_specifications, random_seed=None):
    all_results = defaultdict(lambda :{})
    for args_specification in tqdm(args_specifications):
        args = init_config(args_specification) if random_seed is None else init_config(args_specification+["--seed",str(random_seed)])
        if check_args(args) == False:
            return_dict = {metric:np.nan for metric in metric_names_lm+metric_names_latent} # training not complete
        else:
            return_dict, tp = test_args(args) # test this model
            _, model, _, _ = tp
            # side effect: do text generation and output results into corresponding files
            if model.encoder.useGaussian:
                test_generation(args_specification=None, tp=tp, mode="interpolation")
        all_results[args.dataset][args.model_name] = return_dict
        torch.cuda.empty_cache()
    return all_results

def get_all_analytical_results(args_specifications, random_seeds):
    seeded_results = [get_all_results(args_specifications, random_seed) for random_seed in random_seeds]
    all_analytical_results = {}
    for dataset in seeded_results[0]:
        all_analytical_results[dataset] = {}
        for model_name in seeded_results[0][dataset]:
            all_analytical_results[dataset][model_name] = {}
            for metric_name in seeded_results[0][dataset][model_name]:
                metric_values = [seeded_results[seed][dataset][model_name][metric_name] for seed in range(len(random_seeds))]
                all_analytical_results[dataset][model_name][metric_name] = f"{np.mean(metric_values):.2f} +/- {np.std(metric_values):.2f}"
    return all_analytical_results

def output_all_results_to_excel(all_results, output_path="./evaluation.xlsx", output_metrics=metric_names_lm+metric_names_latent):
    import pandas as pd
    if os.path.exists(output_path):
        output_path = output_path.replace(".xlsx",f".{time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime())}.xlsx")
    pd.DataFrame().to_excel(output_path)
    writer = pd.ExcelWriter(output_path)
    for dataset in all_results:
        model_names = []
        pd_dict = {metric:[] for metric in output_metrics}
        for model_name in all_results[dataset]:
            model_names.append(model_name.split("/")[-1])
            results = all_results[dataset][model_name]
            for metric in output_metrics:
                pd_dict[metric].append(results[metric])
        df = pd.DataFrame(pd_dict, index=model_names)
        df.to_excel(writer, sheet_name=dataset)
        print(df)
    writer.save()
    return

def evaluate_interpolation_results(args_specifications, output_name='unk', num_processes=None):
    from nlg_metrics.static_metrics import bleu, selfbleu, distinct, rouge
    all_results = defaultdict(lambda :{})
    output_metrics = [f"{metric}-{i}" for metric in ["bleu","selfbleu","distinct"] for i in [1,2,3,4]] + [f"rouge-{prf}-{i/10:.1f}" for prf in ["p","r","f"] for i in range(11)]
    def evaluate(args_specification):
        args = init_config(args_specification)
        if not os.path.exists(f"{args.exp_dir}/interpolation.txt"):
            return args, {metric_name:np.nan for metric_name in output_metrics}
        with open(f"{args.exp_dir}/interpolation.txt", "r") as f:
            alllines = f.readlines()
        interpolation_groups = []
        for i in range(int(len(alllines)/14)):
            lines = alllines[i*14+1:(i+1)*14]
            assert "a: <s> " in lines[0] and "b: <s> " in lines[-1]
            sentence_a = lines[0].replace("a: <s> ","").replace(" </s>\n","")
            sentence_b = lines[-1].replace("b: <s> ","").replace(" </s>\n","")
            sentence_inter = [line[6:-6] for line in lines[1:-1]]
            interpolation_groups.append((sentence_a,sentence_b,sentence_inter))
        gts, res = {}, {}
        for i,(sentence_a,sentence_b,sentence_inter) in enumerate(interpolation_groups):
            gts[i*2], gts[i*2+1] = [sentence_a], [sentence_b]
            res[i*2], res[i*2+1] = [sentence_inter[0]], [sentence_inter[-1]]
        bleu_results = bleu(gts,res)
        selfbleu_results = np.mean([selfbleu(sentence_inter) for sentence_a,sentence_b,sentence_inter in interpolation_groups], axis=0)
        distinct_results = np.mean([distinct(sentence_inter) for sentence_a,sentence_b,sentence_inter in interpolation_groups], axis=0)
        
        rouge_results = [rouge(
            [_[0] for _ in interpolation_groups]+[_[1] for _ in interpolation_groups],\
            [_[2][i] for _ in interpolation_groups]*2
        ) for i in range(11)]
        # list[dict[dict]], list in length of 11, dict of ['rouge-1','rouge-2','rouge-l'], dict of ["f","p","r"]
        rouge_results = [_['rouge-l'][prf] for prf in ["p","r","f"] for _ in rouge_results]
        
        return args, {metric_name:metric_value for metric_name,metric_value in zip(
            output_metrics,
            bleu_results + selfbleu_results.tolist() + distinct_results.tolist() + rouge_results
        )}
    if num_processes is None:
        for args_specification in tqdm(args_specifications):
            args, results = evaluate(args_specification)
            if args is not None:
                all_results[args.dataset][args.model_name] = results
    else:
        from multiprocess import Pool
        with Pool(processes=num_processes) as pool:
            args_results = list(tqdm(pool.imap(evaluate, args_specifications), total=len(args_specifications)))
        for args, results in args_results:
            if args is not None:
                all_results[args.dataset][args.model_name] = results
    output_all_results_to_excel(all_results, output_metrics=output_metrics,
                                output_path=f"./evaluation.interpolation.{output_name}.xlsx")
    return

def evaluate_priorsampling_results(args_specifications, output_name='unk', num_processes=None):
    output_metrics = [f"{metric}-{i}" for metric in ["bleu-p","bleu-r","bleu-f"] for i in [1,2,3,4]]
    all_results = defaultdict(lambda :{})
    def evaluate(args_specification, num_piece=50):
        from nlg_metrics.static_metrics import bleu
        args = init_config(args_specification)
        if not os.path.exists(f"{args.exp_dir}/priorsampling.txt"):
            return args, {metric_name:np.nan for metric_name in output_metrics}
        with open(f"{args.exp_dir}/priorsampling.txt", "r") as f:
            alllines = f.readlines()
        gts, res = [], []
        for line in alllines[1:-1]: # the first line is blank, and the last line is end_line
            if line=="gts:\n": out = gts
            elif line=="res:\n": out = res
            else: out.append(line.replace("<s> ","").replace(" </s>\n",""))
        num_piece = min(num_piece, int(np.sqrt(len(res))-1))
        len_piece_gts, len_piece_res = math.ceil(len(gts)/num_piece), math.ceil(len(res)/num_piece)
        def np_weighted_mean(scores, weight):
            return ((scores*weight[:,None]).sum(axis=0) / weight.sum(axis=0)).tolist()
        bleu_p_results = np_weighted_mean(
            scores = np.array([bleu({"all":gts},{"all":res[len_piece_res*i:len_piece_res*(i+1)]}) for i in range(num_piece)]),
            weight = np.array([len(res[len_piece_res*i:len_piece_res*(i+1)]) for i in range(num_piece)]))
        bleu_r_results = np_weighted_mean(
            scores = np.array([bleu({"all":res},{"all":gts[len_piece_gts*i:len_piece_gts*(i+1)]}) for i in range(num_piece)]),
            weight = np.array([len(gts[len_piece_gts*i:len_piece_gts*(i+1)]) for i in range(num_piece)]))
        bleu_f_results = [2*bleu_p*bleu_r/(bleu_p+bleu_r) for bleu_p,bleu_r in zip(bleu_p_results,bleu_r_results)]
        return args, {metric_name:metric_value for metric_name,metric_value in zip(
            output_metrics, bleu_p_results + bleu_r_results + bleu_f_results)}
    if num_processes is None:
        for args_specification in tqdm(args_specifications):
            args, results = evaluate(args_specification)
            all_results[args.dataset][args.model_name] = results
    else:
        from multiprocess import Pool
        with Pool(processes=num_processes) as pool:
            args_results = list(tqdm(pool.imap(evaluate, args_specifications), total=len(args_specifications)))
        for args, results in args_results:
            all_results[args.dataset][args.model_name] = results
    output_all_results_to_excel(all_results, output_metrics=output_metrics,
                                output_path=f"./evaluation.priorsampling.{output_name}.xlsx")
    return

def experiments_evaluation_when_all_complete(args_specifications, random_seeds=None, output_name='unk'):
    if not block_until_all_complete(args_specifications):
        print("abort waiting for experiments to all complete")
    else:
        print("all experiments complete")
    if random_seeds is None:
        all_results = get_all_results(args_specifications)
    else:
        all_results = get_all_analytical_results(args_specifications, random_seeds)
    output_all_results_to_excel(all_results, output_path=f"./evaluation.LM.{output_name}.xlsx")
    # side effect: evaluation for text generation records from corresponding files
    evaluate_interpolation_results(args_specifications, output_name=output_name, num_processes=16)
    #evaluate_priorsampling_results(args_specifications, output_name=output_name, num_processes=16)