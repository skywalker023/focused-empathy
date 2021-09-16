#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Basic example which iterates through the tasks specified and evaluates the given model
on them.

## Examples

```shell
parlai eval_model -t "babi:Task1k:2" -m "repeat_label"
parlai eval_model -t "#CornellMovie" -m "ir_baseline" -mp "-lp 0.5"
```
"""

from parlai.core.params import ParlaiParser, print_announcements
from parlai.core.agents import create_agent
from parlai.core.logs import TensorboardLogger
from parlai.core.metrics import (
    aggregate_named_reports,
    aggregate_unnamed_reports,
    Metric,
)
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger, nice_report
from parlai.utils.world_logging import WorldLogger
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.io import PathManager
import parlai.utils.logging as logging
from parlai.core.metrics import AverageMetric

import os
import json
import random
from tqdm import tqdm
from copy import deepcopy
import torch

from parlai.utils.distributed import (
    is_primary_worker,
    all_gather_list,
    is_distributed,
    get_rank,
)

from modules.empathy_scorer import EmpathyScorer
from zoo.blender.build import download as download_blender

def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Evaluate a model')
    # Get command line arguments
    parser.add_argument(
        '-rf',
        '--report-filename',
        type=str,
        default='',
        help='Saves a json file of the evaluation report either as an '
        'extension to the model-file (if begins with a ".") or a whole '
        'file path. Set to the empty string to not save at all.',
    )
    parser.add_argument(
        '--save-world-logs',
        type='bool',
        default=False,
        help='Saves a jsonl file containing all of the task examples and '
        'model replies. Must also specify --report-filename.',
    )
    parser.add_argument(
        '--save-format',
        type=str,
        default='conversations',
        choices=['conversations', 'parlai'],
    )
    parser.add_argument('-ne', '--num-examples', type=int, default=-1)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument(
        '--display-add-fields',
        type=str,
        default='full_text',
        help='Specify extra fields for display'
    )
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=10)
    parser.add_argument(
        '-mcs',
        '--metrics',
        type=str,
        default='default',
        help='list of metrics to show/compute, e.g. all, default,'
        'or give a list split by , like '
        'ppl,f1,accuracy,hits@1,rouge,bleu'
        'the rouge metrics will be computed as rouge-1, rouge-2 and rouge-l',
    )
    parser.add_argument(
        '-micro',
        '--aggregate-micro',
        type='bool',
        default=False,
        help='Report micro-averaged metrics instead of macro averaged metrics.',
        recommended=False,
    )
    parser.add_argument(
        '--empathy-score',
        type='bool',
        default=False,
        help="Compute empathy identification score if true (please see the paper {A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support} for more details)",
    )
    WorldLogger.add_cmdline_args(parser)
    TensorboardLogger.add_cmdline_args(parser)
    parser.set_params(datatype='test')
    return parser


def _save_eval_stats(opt, report):
    if not is_primary_worker:
        return
    report_fname = opt['report_filename']
    if report_fname == '':
        return

    json_serializable_report = report
    for k, v in report.items():
        if isinstance(v, Metric):
            v = v.value()
        json_serializable_report[k] = v

    # Save report
    with PathManager.open(report_fname, 'w') as f:
        logging.info(f'Saving model report to {report_fname}')
        json.dump({'opt': opt, 'report': json_serializable_report}, f, indent=4)
        f.write("\n")  # for jq


def _eval_single_world(
    opt, agent, task,
    epitome_empathy_scorer: EmpathyScorer =None,
):
    logging.info(f'Evaluating task {task} using datatype {opt.get("datatype")}.')
    # set up world logger
    world_logger = WorldLogger(opt) if opt['save_world_logs'] else None

    task_opt = opt.copy()  # copy opt since we're editing the task
    task_opt['task'] = task
    world = create_task(task_opt, agent)  # create worlds for tasks

    # set up logging
    log_every_n_secs = opt.get('log_every_n_secs', -1)
    if log_every_n_secs <= 0:
        log_every_n_secs = float('inf')
    log_time = TimeLogger()

    # max number of examples to evaluate
    max_cnt = opt['num_examples'] if opt['num_examples'] > 0 else float('inf')
    pbar_desc = "Generation"
    if max_cnt == float('inf'):
        try:
            pbar = tqdm(total=world.worlds[0].agents[0].num_examples(), desc=pbar_desc)
        except AttributeError as e:
            pbar = tqdm(total=world.agents[0].num_examples(), desc=pbar_desc)
    else:
        pbar = tqdm(total=max_cnt, desc=pbar_desc)
    cnt = 0
    total_cnt = world.num_examples()

    dump_dir = os.path.dirname(opt['report_filename'])
    if dump_dir != '':
        os.makedirs(dump_dir, exist_ok=True)
        report_filename = opt['report_filename'].split('/')[-1]
        if opt.get('num_generation_split', 1) > 1:
            current_split = opt['current_split']
            dump_fname = os.path.join(
                dump_dir, f'{report_filename}_dump_split{current_split}.jsonl')
        else:
            dump_fname = os.path.join(dump_dir, f'{report_filename}_dump.jsonl')
        dump_fp = open(dump_fname, 'w')

    if is_distributed():
        logging.warn('Progress bar is approximate in distributed mode.')

    if epitome_empathy_scorer:
        IP_scores = []
        EX_scores = []

    while not world.epoch_done() and cnt < max_cnt:
        cnt += opt.get('batchsize', 1)
        world.parley()
        if world_logger is not None:
            world_logger.log(world)
        if opt['display_examples'] and cnt % (50 * opt.get('batchsize')) == 0:
            # display examples
            print(world.display() + '\n~~')
        if log_time.time() > log_every_n_secs:
            report = world.report()
            text, report = log_time.log(
                report.get('exs', 0), min(max_cnt, total_cnt), report
            )
            logging.info(text)
        pbar.update(cnt - pbar.n)

        # Prepare input for computing empathy score
        if epitome_empathy_scorer:
            seeker_posts = []
            response_posts = []
            for w in world.worlds:
                teacher_act = w.acts[0]
                agent_act = w.acts[1]

                seeker_post = teacher_act.get('text', None)
                response_post = agent_act.get('text', None)

                if seeker_post is None or response_post is None:
                    continue

                seeker_posts.append(seeker_post)
                response_posts.append(response_post)

        # Compute empathy score
        if epitome_empathy_scorer and len(seeker_posts) > 0:
            empathy_scores = epitome_empathy_scorer(seeker_posts, response_posts)
            IP_scores += empathy_scores['IP'][0]
            EX_scores += empathy_scores['EX'][0]

        for w in world.worlds:
            teacher_act = w.acts[0]
            agent_act = deepcopy(w.acts[1])
            try:
                for key in agent_act['metrics'].keys():
                    agent_act['metrics'][key] = agent_act['metrics'][key].value()
            except KeyError as e:
                # Invalid batch
                continue
            all_acts = {'teacher': teacher_act, 'agent': agent_act}
            if dump_dir != '':
                dump_fp.write(json.dumps(all_acts) + '\n')
                dump_fp.flush()
    pbar.close()
    if dump_dir != '':
        dump_fp.close()

    if world_logger is not None:
        # dump world acts to file
        world_logger.reset()  # add final acts to logs
        base_outfile = opt['report_filename']
        if is_distributed():
            rank = get_rank()
            outfile = base_outfile + f'_{task}_{rank}_replies.jsonl'
        else:
            outfile = base_outfile + f'_{task}_replies.jsonl'
        world_logger.write(outfile, world, file_format=opt['save_format'])

    _report = world.report()
    if epitome_empathy_scorer:
        _report['empathy_epitome_IP'] = AverageMetric(sum(IP_scores), len(IP_scores))
        _report['empathy_epitome_EX'] = AverageMetric(sum(EX_scores), len(EX_scores))

    report = aggregate_unnamed_reports(all_gather_list(_report))
    world.reset()

    return report


def eval_model(opt):
    """
    Evaluates a model.

    :param opt: tells the evaluation function how to run
    :return: the final result of calling report()
    """
    device = list(range(torch.cuda.device_count()))[-1]

    random.seed(42)
    if 'train' in opt['datatype'] and 'evalmode' not in opt['datatype']:
        raise ValueError(
            'You should use --datatype train:evalmode if you want to evaluate on '
            'the training set.'
        )

    if opt['save_world_logs'] and not opt['report_filename']:
        raise RuntimeError(
            'In order to save model replies, please specify the save path '
            'with --report-filename'
        )

    epitome_empathy_scorer = None
    if opt['empathy_score']:
        epitome_empathy_scorer = EmpathyScorer(opt, batch_size=1, cuda_device=device)

    if 'Blender' in opt['model']:
        download_blender(opt['datapath'])

    # load model and possibly print opt
    agent = create_agent(opt, requireModelExists=True)
    agent.opt.log()

    tasks = opt['task'].split(',')
    reports = []
    for task in tasks:
        task_report = _eval_single_world(opt, agent, task,
                                         epitome_empathy_scorer)
        reports.append(task_report)

    report = aggregate_named_reports(
        dict(zip(tasks, reports)), micro_average=opt.get('aggregate_micro', False)
    )

    # print announcments and report
    print_announcements(opt)
    logging.info(
        f'Finished evaluating tasks {tasks} using datatype {opt.get("datatype")}'
    )

    print(nice_report(report))
    _save_eval_stats(opt, report)
    return report


@register_script('eval_model', aliases=['em', 'eval'])
class EvalModel(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return eval_model(self.opt)


if __name__ == '__main__':
    EvalModel.main()
