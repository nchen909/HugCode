import logging
import torch
import argparse
import time
import multiprocessing
import os
import numpy as np
import math
import json
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from configs import add_args, set_dist, set_seed, set_hyperparas
from models import bulid_or_load_gen_model,bulid_or_load_cls_model
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data, load_and_cache_defect_data,load_and_cache_clone_data, get_lang_by_task
from metrics import smooth_bleu
from metrics.CodeBLEU import calc_code_bleu
from metrics.bleu import _bleu
import sys
from sklearn.metrics import recall_score, precision_score, f1_score
from tree_sitter import Language, Parser
from utils import retrieve2file
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_cls(args, model, eval_examples, eval_data, write_to_pred=False):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.dev_batch_size)

    # Eval!
    if write_to_pred == False:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Num batches = %d", len(eval_dataloader))
        logger.info("  Batch size = %d", args.dev_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating"):
        if args.sub_task == "POJ":
            inputs = batch[0].to(args.device)    
            p_inputs = batch[1].to(args.device)
            n_inputs = batch[2].to(args.device)
            label = batch[3].to(args.device)
        else:
            inputs = batch[0].to(args.device) # inputs shape: [batch_size, args.max_source_length+args.max_target_length]
            label = batch[1].to(args.device) # label shape: [batch_size]
        with torch.no_grad():
            if args.sub_task == "POJ":
                lm_loss, logit = model(inputs,p_inputs,n_inputs,label)
            else:
                lm_loss, logit = model(inputs, label) # logit shape:[batch_size]
            
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())#logit shape: [batch_size]
            labels.append(label.cpu().numpy())#label shape: [batch_size]
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    if args.model_name == "unixcoder" and args.task == "clone":
        preds = logits > 0.5
    else:
        preds = logits[:, 1] > 0.5
    if args.task == 'defect':
        eval_acc = np.mean(labels == preds)
        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.tensor(eval_loss)

        result = {
            "eval_loss": float(perplexity),
            "eval_acc": round(eval_acc, 4) * 100,
        }
    elif args.task == 'clone':
        recall = recall_score(labels, preds)
        precision = precision_score(labels, preds)
        f1 = f1_score(labels, preds)
        result = {
            "eval_recall": float(recall) * 100,
            "eval_precision": float(precision) * 100,
            "eval_f1": float(f1) * 100,
            "eval_threshold": 0.5,
        }
    if write_to_pred == False:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    if write_to_pred:
        with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
            for example, pred in zip(eval_examples, preds):
                if args.task == 'defect':
                    if args.model_name == "unixcoder":
                        if pred:
                            f.write(str(example.idx) + '\t1\n')
                        else:
                            f.write(str(example.idx) + '\t0\n')
                    else:
                        if pred:
                            f.write(str(example.idx) + '\t1\n')
                        else:
                            f.write(str(example.idx) + '\t0\n')
                elif args.task == 'clone':
                    if pred:
                        f.write(example.url1 + '\t' + example.url2 + '\t' + '1' + '\n')
                    else:
                        f.write(example.url1 + '\t' + example.url2 + '\t' + '0' + '\n')

    return result