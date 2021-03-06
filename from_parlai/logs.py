#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Log metrics to tensorboard.

This file provides interface to log any metrics in tensorboard, could be
extended to any other tool like visdom.

.. code-block: none

   tensorboard --logdir <PARLAI_DATA/tensorboard> --port 8888.
"""

import json
import numbers
from parlai.core.opt import Opt
from parlai.core.metrics import Metric
from parlai.utils.io import PathManager
import parlai.utils.logging as logging

MISC_METRICS = ['tpb', 'gnorm', 'clip', 'updates', 'lr', 'gpu_mem',
                'total_train_updates']


class TensorboardLogger(object):
    """
    Log objects to tensorboard.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        """
        Add tensorboard CLI args.
        """
        logger = argparser.add_argument_group('Tensorboard Arguments')
        logger.add_argument(
            '-tblog',
            '--tensorboard-log',
            type='bool',
            default=False,
            help="Tensorboard logging of metrics, default is %(default)s",
            hidden=False,
        )
        logger.add_argument(
            '-tblogdir',
            '--tensorboard-logdir',
            type=str,
            default=None,
            help="Tensorboard logging directory, defaults to model_file.tensorboard",
            hidden=False,
        )

    def __init__(self, opt: Opt):
        try:
            # tensorboard is a very expensive thing to import. Wait until the
            # last second to import it.
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError('Please run `pip install tensorboard tensorboardX`.')

        if opt['tensorboard_logdir'] is not None:
            tbpath = opt['tensorboard_logdir']
        else:
            tbpath = opt['model_file'] + '.tensorboard'

        logging.debug(f'Saving tensorboard logs to: {tbpath}')
        if not PathManager.exists(tbpath):
            PathManager.mkdirs(tbpath)
        self.writer = SummaryWriter(tbpath, comment=json.dumps(opt))

    def log_metrics(self, setting, step, report):
        """
        Add all metrics from tensorboard_metrics opt key.

        :param setting:
            One of train/valid/test. Will be used as the title for the graph.
        :param step:
            Number of parleys
        :param report:
            The report to log
        """
        for k, v in report.items():
            # Manually convert colon (:) to underbar (_) to avoid warning
            k = k.replace(':', '_')

            # Manually categorize global metrics and misc metrics
            if k.startswith("tasks."):
                pass
            elif k in MISC_METRICS:
                k = f"misc/{k}"
            else:
                k = f"macro/{k}"

            if isinstance(v, numbers.Number):
                #self.writer.add_scalar(f'{k}/{setting}', v, global_step=step)
                self.writer.add_scalar(f'{setting}_{k}', v, global_step=step)
            elif isinstance(v, Metric):
                #self.writer.add_scalar(f'{k}/{setting}', v.value(), global_step=step)
                self.writer.add_scalar(f'{setting}_{k}', v.value(), global_step=step)
            else:
                logging.error(f'k {k} v {v} is not a number')

    def flush(self):
        self.writer.flush()
