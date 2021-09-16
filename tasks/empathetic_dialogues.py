#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from typing import Any, List

import numpy as np

from parlai.utils.io import PathManager
from parlai.core.teachers import FixedDialogTeacher
from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
from utils.etc_utils import set_random_seed

DEFAULT_TRAIN_EXPERIENCER_ONLY = True
DEFAULT_REMOVE_POLITICAL_CONVOS = False

RESOURCES = [
    DownloadableFile(
        '10md_XMkzMxDWlN3hKi-vYqofyMcNFtLx',
        'json_empatheticdialogues.tar.gz',
        'fb6c40e257c838382aa216a1f2ea70aa3f4eae0f78f5948dc3336f19ebde2406',
        zipped=True, from_google=True,
    )
]

def _build(opt):
    dpath = os.path.join(opt['datapath'], 'empatheticdialogues')
    # Update log
    # v1.0: empathetic dialogues + situation_rake_keywords
    # v1.1: v1.0 + utterance_rake_keywords
    version = '1.1'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)

class EmpatheticDialoguesTeacher(FixedDialogTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt
        base_datatype = self.datatype.split(':')[0]
        self.datapath = os.path.join(
            self.opt['datapath'],
            'empatheticdialogues',
            base_datatype + '.json',
        )
        self.experiencer_side_only = (
            opt.get('train_experiencer_only', DEFAULT_TRAIN_EXPERIENCER_ONLY)
            and base_datatype == 'train'
        ) or base_datatype != 'train'
        if not shared:
            print(
                f'[EmpatheticDialoguesTeacher] Only use experiencer side? '
                f'{self.experiencer_side_only}, datatype: {self.datatype}'
            )
        self.remove_political_convos = opt.get(
            'remove_political_convos', DEFAULT_REMOVE_POLITICAL_CONVOS
        )

        if shared:
            self.data = shared['data']
        else:
            _build(opt)
            self._setup_data(base_datatype)

        seed = opt.get('random_seed', 33)
        set_random_seed(seed)  # Set random seed after setup_data not to be ignored by random.seed in setup_data
        self.num_exs = sum([len(d) for d in self.data])
        self.num_eps = len(self.data)
        self.reset()

    @classmethod
    def add_cmdline_args(cls, argparser):
        agent = argparser.add_argument_group('EmpatheticDialogues teacher arguments')
        agent.add_argument(
            '--train-experiencer-only',
            type='bool',
            default=DEFAULT_TRAIN_EXPERIENCER_ONLY,
            # i.e. do not include the other side of the conversation where the Listener
            # (responder) utterance would be the text and the Speaker (experiencer)
            # utterance would be the label
            help='In the train set, only use Speaker (experiencer) utterances as text and Listener (responder) utterances as labels.',
        )
        agent.add_argument(
            '--remove-political-convos',
            type='bool',
            default=DEFAULT_REMOVE_POLITICAL_CONVOS,
            help='Remove all conversations containing an utterance marked as political',
        )
        agent.add_argument(
            '--random-seed',
            type=int,
            default=33
        )

    def num_episodes(self):
        return self.num_eps

    def num_examples(self):
        return self.num_exs

    def _setup_data(self, base_datatype):
        """
        self.data:
            [dialogue1, dialogue2, ..., dialogueN]
        dialogue:
            [[context, utterance, emotion, situation, ...], [c, u, e, s, ...], ..., [c,u,e,s,..]]
        """

        if self.opt.get('deepmoji') is not None:
            self.embed = np.load(self.opt['deepmoji'] + base_datatype + ".npy")

        if self.opt.get('fasttextloc') is not None and self.opt.get('prepend', -1) > 0:
            try:
                import fastText
            except ImportError:
                raise ImportError("Please run 'pip install fasttext'.")
            ftpath = self.opt['fasttextloc']
            ftmodel = fastText.FastText.load_model(ftpath)

        with PathManager.open(self.datapath) as f:
            df = json.load(f)

        turn_idx = 1
        responder_text_dialogue = []
        experiencer_text_dialogue = []
        self.data = []
        situations = []

        for i in range(1, len(df)):
            cparts = df[i - 1]  # instance that will act as context
            sparts = df[i]  # intance that will act as response sentence

            if cparts['conv_id'] == sparts['conv_id']:

                # Check that the turn number has incremented correctly
                turn_idx += 1
                assert (
                    int(cparts['utterance_idx']) + 1 == int(sparts['utterance_idx']) and int(sparts['utterance_idx']) == turn_idx
                )

                contextt = cparts['utterance']
                label = sparts['utterance']
                emotion = sparts['emotion']
                situation = sparts['situation']
                situation_keywords = sparts['situation_rake_keywords']  # list of keywords

                # label candidates partially exist in valid and test set
                if len(sparts) == 11:
                    if sparts['inline_label_cands'] != '':
                        inline_label_candidates = sparts['inline_label_cands']
                    else:
                        inline_label_candidates = []
                elif len(sparts) == 10:
                    inline_label_candidates = []
                else:
                    raise ValueError(f'Line {i:d} has the wrong number of fields!')

                context_emb, cand_emb = None, None
                if self.opt.get('deepmoji') is not None:
                    context_emb = self.embed[i - 2]
                    cand_emb = self.embed[i - 1]

                ft_ctx, ft_cand = None, None
                if (
                    self.opt.get('fasttextloc') is not None
                    and self.opt.get('prepend', -1) > 0
                ):
                    ft_ctx = ""
                    gettop, _ = ftmodel.predict(contextt, k=self.opt['prepend'])
                    for f in gettop:
                        ft_ctx = f.split("_")[-1] + " " + ft_ctx
                    ft_cand = ""
                    gettop, _ = ftmodel.predict(label, k=self.opt['prepend'])
                    for f in gettop:
                        ft_cand = f.split("_")[-1] + " " + ft_cand

                # Check if either the text or label are marked as being political
                is_political = '<POLITICAL>' in cparts['tags'] or '<POLITICAL>' in sparts['tags']

                dialogue_parts = [
                    contextt,
                    label,
                    emotion,
                    situation,
                    context_emb,
                    cand_emb,
                    ft_ctx,
                    ft_cand,
                    inline_label_candidates,
                    is_political,
                    situation_keywords,
                ]

                if int(sparts['utterance_idx']) % 2 == 0:
                    # experiencer is the "text" and responder is the "label"
                    experiencer_text_dialogue.append(dialogue_parts)
                else:
                    # responder is the "text" and experiencer is the "label"
                    responder_text_dialogue.append(dialogue_parts)

            else:
                # aggregate average keyword count in situation
                situations.append(len(df[i-1]['situation_rake_keywords']))

                # We've finished the previous episode, so add it to the data
                turn_idx = 1
                self.data += self._select_dialogues_to_add(
                    experiencer_text_dialogue, responder_text_dialogue
                )
                experiencer_text_dialogue = []
                responder_text_dialogue = []

        # Add in the final episode
        self.data += self._select_dialogues_to_add(
            experiencer_text_dialogue, responder_text_dialogue
        )

    def _select_dialogues_to_add(
        self,
        experiencer_text_dialogue: List[List[Any]],
        responder_text_dialogue: List[List[Any]],
    ) -> List[List[List[Any]]]:
        """
        Return conversation halves to add to self.data.
        Given lists corresponding to the conversation turns from both sides of the
        conversation, return only the list(s) that will be added to self.data.
        Optionally filter by side of the conversation or by whether the conversation
        contains any political language.
        """

        # experiencer_dialogue:
        #   - context: speaker talking about the emotional situtation
        #   - utterance: empathetic response
        # responder_text_dialogue:
        #   - context: empathetic response
        #   - utterance: speaker talking about the emotional situation

        if self.remove_political_convos and any(
            [turn[9] for turn in experiencer_text_dialogue + responder_text_dialogue]
        ):
            return []
        else:
            selected_dialogues = []
            if len(experiencer_text_dialogue) > 0:
                selected_dialogues.append(experiencer_text_dialogue)
            if len(responder_text_dialogue) > 0 and not self.experiencer_side_only:
                selected_dialogues.append(responder_text_dialogue)
            return selected_dialogues

    def get(self, episode_idx, entry_idx=0):
        ep = self.data[episode_idx]
        ep_i = ep[entry_idx]
        episode_done = entry_idx >= (len(ep) - 1)
        action = {
            'situation': ep_i[3],
            'emotion': ep_i[2],
            'text': ep_i[0],
            'labels': [ep_i[1]],
            'prepend_ctx': ep_i[6],
            'prepend_cand': ep_i[7],
            'deepmoji_ctx': ep_i[4],
            'deepmoji_cand': ep_i[5],
            'episode_done': episode_done,
            'label_candidates': ep_i[8],
            'situation_rake_keywords': ep_i[9],
        }
        return action

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared

class DefaultTeacher(EmpatheticDialoguesTeacher):
    pass
