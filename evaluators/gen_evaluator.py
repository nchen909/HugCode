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


def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.dev_batch_size,
                                 num_workers=4, pin_memory=True)
    # Start evaluating model
    logger.info("  " + "***** Running ppl evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.dev_batch_size)

    model.eval()
    eval_loss, batch_num = 0, 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval ppl"):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_name in ['roberta', 'codebert', 'graphcodebert']:
#                     loss, _, _, attention = model(source_ids=source_ids, source_mask=source_mask,
#                                           target_ids=target_ids, target_mask=target_mask)
                loss, _, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                    target_ids=target_ids, target_mask=target_mask)
                if args.n_gpu > 1:
                    loss = loss.mean()
                eval_loss += loss.item()
                batch_num += 1
            elif args.model_name in ['unixcoder']:
                _,loss,num = model(source_ids=source_ids,target_ids=target_ids)
                if args.n_gpu > 1:
                    loss = loss.mean()
                eval_loss += loss.sum().item()
                batch_num += num.sum().item()
            else:
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                if isinstance(outputs,dict):
                    loss=outputs['loss']
                else:
                    loss = outputs.loss
                if args.n_gpu > 1:
                    loss = loss.mean()
                eval_loss += loss.item()
                batch_num += 1
    
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl


def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    logger.info(
        "  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    if split_tag == 'dev':
        batch_size = args.dev_batch_size
    elif split_tag == 'test':
        batch_size = args.test_batch_size
    else:
        batch_size = args.batch_size
    logger.info("  Batch size = %d", batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=batch_size)

    model.eval()
    pred_ids = []
    bleu, codebleu = 0.0, 0.0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        source_ids = batch[0].to(args.device) #shape: (batch_size, max_source_len)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        if hasattr(model, 'module'):
            model = model.module # extract the model from the DataParallel wrapper
        with torch.no_grad():
            if args.model_name in ['roberta', 'codebert', 'graphcodebert']:
                preds, _ = model(source_ids=source_ids,
                                 source_mask=source_mask)
                top_preds = [pred[0].cpu().numpy() for pred in preds]
            elif args.model_name in ['unixcoder']:
                preds = model(source_ids=source_ids)  # preds shape: [batch_size, self.beam_size, max_target_len]
                top_preds = [pred[0].cpu().numpy() for pred in preds]# top_preds shape: batch_size * [max_target_len]
            else:
                preds = model.generate(source_ids,
                                       attention_mask=source_mask,
                                       use_cache=True,
                                       num_beams=args.beam_size,
                                       early_stopping=args.task == 'summarize',
                                       max_length=args.max_target_length)
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)
    # pdb.set_trace()
    pred_nls = [tokenizer.decode(
        id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
    # unixcoder in fewshot will generate '\n' in small batch, and gradually disappear
    if args.model_name in ['unixcoder']:
        pred_nls = [id.replace('\n',' ').replace("        "," ").replace("    "," ").replace("\t"," ") for id in pred_nls]
    output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
    gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
    src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))

    if args.task in ['defect']:
        target_dict = {0: 'false', 1: 'true'}
        golds = [target_dict[ex.target] for ex in eval_examples]
        eval_acc = np.mean([int(p == g) for p, g in zip(pred_nls, golds)])
        result = {'em': eval_acc, 'bleu': 0, 'codebleu': 0}

        with open(output_fn, 'w',encoding='utf-8') as f, open(gold_fn, 'w',encoding='utf-8') as f1, open(src_fn, 'w',encoding='utf-8') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                f.write(pred_nl.strip() + '\n')
                f1.write(target_dict[gold.target] + '\n')
                f2.write(gold.source.strip() + '\n')
            logger.info("Save the predictions into %s", output_fn)
    else:
        dev_accs, predictions = [], []
        with open(output_fn, 'w',encoding='utf-8') as f, open(gold_fn, 'w',encoding='utf-8') as f1, open(src_fn, 'w',encoding='utf-8') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                dev_accs.append(pred_nl.strip() == gold.target.strip())
                if args.task in ['summarize']:
                    predictions.append(str(gold.idx) + '\t' + pred_nl)
                    f.write(str(gold.idx) + '\t' +
                            pred_nl.strip().encode('utf8').decode() + '\n')
                    f1.write(str(gold.idx) + '\t' +
                             gold.target.strip().encode('utf8').decode() + '\n')
                    f2.write(str(gold.idx) + '\t' +
                             gold.source.strip().encode('utf8').decode() + '\n')
                else:
                    f.write(pred_nl.strip().encode('utf8').decode() + '\n')
                    f1.write(gold.target.strip().encode(
                        'utf8').decode() + '\n')
                    f2.write(gold.source.strip().encode(
                        'utf8').decode() + '\n')
        if args.task in ['summarize']:
            (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
            bleu = round(smooth_bleu.bleuFromMaps(
                goldMap, predictionMap)[0], 2)
        else:
            bleu = round(_bleu(gold_fn, output_fn), 2)
            if split_tag == 'test' and args.task in ['refine', 'translate', 'generate' , 'clone']:
                args.lang = get_lang_by_task(args.task, args.sub_task)
                codebleu = calc_code_bleu.get_codebleu(
                    gold_fn, output_fn, args.lang,args=args)
        # except:
        #     bleu = 0.0
        #     codebleu = 0.0

        em = np.mean(dev_accs) * 100
        result = {'em': em, 'bleu': bleu}
        if not args.task == 'summarize' and split_tag == 'test':
            result['codebleu'] = codebleu * 100

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result