import os
from itertools import cycle
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F

from parlai.core.opt import Opt
from parlai.core.message import Message
from parlai.core.torch_agent import Batch, Output
from parlai.core.torch_generator_agent import PPLMetric
from parlai.core.metrics import AverageMetric
from parlai.utils.torch import padded_tensor
from parlai.utils.misc import warn_once
from parlai.agents.transformer.transformer import (
    TransformerGeneratorAgent,
    add_common_cmdline_args
)

from zoo.gee.build import download as gee_download
from modules.gee_module import GeeModel
from modules.pragmatic_blender_module import PragmaticTransformerModel
from agents.history import EmpatheticHistory
from utils.etc_utils import EMOTION_LABELS


class EmpatheticBlenderAgent(TransformerGeneratorAgent):
    """
    Implementation of the Empathetic Blender Agent.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        group = argparser.add_argument_group('Empathetic Blender Arguments')
        group.add_argument(
            '-histsz',
            '--history-size',
            default=2,
            type=int,
            help='Number of past dialog utterances to remember.',
        )
        group.add_argument(
            '--emotion-classes',
            type=str,
            nargs='*',
            default=EMOTION_LABELS,
            help='Emotion class labels'
        )
        group.add_argument(
            '--update-emotion-prior',
            type=bool,
            default=False,
            help="Update emotion prior with previous turn's posterior"
        )
        group.add_argument(
            '--topk-causes',
            type=int,
            default=5,
            help='Specify number of causes to use'
        )
        group.add_argument(
            '--pragmatic-target',
            type=str,
            choices=['none', 'previous_utterance'],
            default='previous_utterance',
            help='The target which the agent will be pragmatic on',
        )
        group.add_argument(
            '-a',
            '--alpha',
            type=float,
            default=6,
            help='Rationality parameter for S_1(speaker_1)',
        )
        group.add_argument(
            '-b',
            '--beta',
            type=float,
            default=0.9,
            help='Rationality parameter for Listener',
        )
        group.add_argument(
            '--world-cardinality',
            type=int,
            default=3,
            help='Size of world for RSA (ground truth + #distractors)'
        )
        group.add_argument(
            '--worldprior',
            type=str,
            choices=['uniform', 'L0', 'L1'],
            default='L0',
            help='Update world prior with a `uniform` distribution or `L0` or `L1`.',
        )
        group.add_argument(
            '--distractor-type',
            type=str,
            choices=['gee-focused', 'random'],
            default='gee-focused',
            help='Specify which distractor type to use'
        )
        add_common_cmdline_args(group)
        cls.dictionary_class().add_cmdline_args(argparser)

        super(EmpatheticBlenderAgent, cls).add_cmdline_args(argparser)
        return group

    def __init__(self, opt: Opt, shared=None):

        self.task = str.lower(opt['task'].split(':')[-1])

        self.available_devices = cycle(range(torch.cuda.device_count()))
        opt['gpu'] = next(self.available_devices)

        super().__init__(opt, shared)

        self._build_gee(shared, use_cuda=True)
        self.emotion_classes = opt.get('emotion_classes')
        self.emo_cardinality = len(self.emotion_classes)
        self.topk_causes = opt.get('topk_causes', 5)

        # the pragmatic target utterance will always be at index 0
        self.pragmatic_target = opt.get('pragmatic_target', 'previous_utterance')
        self.world_cardinality = opt.get('world_cardinality', 3)
        self.alpha = 0.0 if self.pragmatic_target == 'none' else opt.get('alpha', 6.0)
        self.beta = opt.get('beta', 0.9)
        self.worldprior = opt.get('worldprior', 'L0')
        self.distractor_type = opt.get('distractor_type', 'gee-focused')

        self.eval_type = opt.get('eval_type')
        self.rank_candidates = opt.get('rank_candidates', False)
        self.multigpu = (
            opt.get('multigpu', False) and self.use_cuda and (opt.get('batchsize') > 1)
        )

        # Implementation is based on beam_size 1
        self.beam_size = 1
        warn_once(f'This implementation is assumed to have beam-size 1.')

        self.id = 'EmpatheticBlender'

        self.reset()

    def _build_gee(self, shared, use_cuda=True):
        gee_dir = gee_download(self.opt['datapath'])
        self.gee_gpu = next(self.available_devices)

        if not shared:
            self.gee_opt = Opt.load(os.path.join(gee_dir, 'model.opt'))
            self.gee_opt['gee_checkpoint'] = os.path.join(gee_dir, 'model')
            self.gee_opt['dict_file'] = os.path.join(gee_dir, 'model.dict')
            self.gee_opt['datapath'] = self.opt.get('datapath')

            dictionary = self.dictionary_class()(self.gee_opt)
            special_toks = self._get_special_tokens()
            if special_toks:
                dictionary.add_additional_special_tokens(special_toks)

            if self.opt.get('person_tokens'):
                dictionary[self.P1_TOKEN] = 999_999_999
                dictionary[self.P2_TOKEN] = 999_999_998

            self.gee_dict = dictionary
            extra_opts = ['emotion_classes', 'topk_causes', 'world_cardinality', 'text_truncate']

            for k in extra_opts:
                self.gee_opt[k] = self.opt[k]
            gee = GeeModel(self.gee_opt, self.gee_dict, device=self.gee_gpu)
            gee.eval()
            if use_cuda:
                gee.cuda(self.gee_gpu)
            self.gee = gee
        else:
            self.gee = shared['gee']

    def build_model(self, states=None) -> PragmaticTransformerModel:
        """
        Build and return model.
        """
        model = PragmaticTransformerModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model

    def history_class(self):
        return EmpatheticHistory

    def _pad_tensor(
        self, items: List[Union[List[int], torch.LongTensor]], device=None
    ) -> Tuple[torch.LongTensor, List[int]]:
        """
        Create a right padded matrix from an uneven list of lists.

        Returns (padded, lengths), where padded is the padded matrix, and lengths
        is a list containing the lengths of each row.

        :param list[iter[int]] items: List of items
        :returns: (padded, lengths) tuple
        :rtype: (Tensor[int64], list[int])

        This is intentionally overridable so that models can control how
        to pad their input.
        """
        cuda_device = self.opt['gpu'] if device is None else device
        return padded_tensor(
            items,
            pad_idx=self.NULL_IDX,
            use_cuda=self.use_cuda,
            fp16friendly=self.opt['fp16'],
            device=cuda_device,
        )

    def _model_input(self, batch):
        """
        Override from TorchGeneratorAgent
        passes (batch.text_vec,) to TorchGeneratorAgent._encoder_input()
        TGA._encoder_input() directly passes the result of TGA._model_input()
        change batch.text_vec to batch.distractor_text_vec for pragmatic decoding
        """
        return (batch.distractor_vec,)

    def pragmatic_greedy_generate(self, batch, maxlen):
        """
        Greedy decoding with pragmatics
        """
        encoder_states = self.model.encoder(*self._encoder_input(batch))
        preds, scores = self.model.pragmatic_decode(encoder_states, maxlen)

        return preds, scores

    def rank(self, batch):
        """
        Rank candidates by PPL score
        """
        bsz = batch.text_vec.size(0)
        world_cardinality = self.world_cardinality
        embedding_size = self.opt.get('embedding_size')
        ranked_candidates = []
        cand_ordering = []
        encoder_states = self.model.encoder(*self._encoder_input(batch))
        batch_dim = encoder_states[0].size(0)  # two possibilities: batchsize or batchsize * world_cardinality

        if bsz != batch_dim:
            enc_output = encoder_states[0].view(bsz, world_cardinality, -1, embedding_size).contiguous()
            enc_output_mask = encoder_states[1].view(bsz, world_cardinality, -1).contiguous()
            encoder_states = (enc_output, enc_output_mask)

        for i in range(bsz):
            num_cands = len(batch.candidate_vecs[i])
            cands, _ = self._pad_tensor(batch.candidate_vecs[i])
            # get [i]th state from encoder_states #num_cands time.
            # because we need same encoder_states for each candidate
            enc = self.model.reorder_encoder_states(encoder_states, [i] * num_cands)

            # enc: (num_cands, world_cardinality, seqlen, emb_size)
            # scores: (num_cands, max_len, vocab_size)
            scores, _ = self.model.pragmatic_decode_forced(enc, cands)

            cand_losses = F.cross_entropy(
                scores.view(num_cands * cands.size(1), -1),
                cands.view(-1),
                reduction='none',
            ).view(num_cands, cands.size(1))
            # now cand_losses is cands x seqlen size, but we still need to
            # check padding and such
            mask = (cands != self.NULL_IDX)
            mask = mask.half() if self.fp16 else mask.float()
            cand_scores = (-cand_losses * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)

            _, ordering = cand_scores.sort(descending=True)
            ranked_candidates.append([batch.candidates[i][o] for o in ordering])
            cand_ordering.append(ordering)

        return ranked_candidates, cand_ordering

    def compute_loss(self, batch, return_output=False):
        """
        Override from TorchGeneratorAgent
        Compute and return the loss for the given batch.

        Easily overridable for customized loss functions.

        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')

        bsz = batch.text_vec.size(0)
        world_cardinality = self.world_cardinality
        embedding_size = self.opt.get('embedding_size')
        encoder_states = self.model.encoder(*self._encoder_input(batch))

        enc_output = encoder_states[0].view(bsz, world_cardinality, -1, embedding_size).contiguous()
        enc_output_mask = encoder_states[1].view(bsz, world_cardinality, -1).contiguous()
        encoder_states = (enc_output, enc_output_mask)

        scores, preds = self.model.pragmatic_decode_forced(encoder_states, batch.label_vec)
        model_output = (scores, preds, encoder_states)

        score_view = scores.view(-1, scores.size(-1))
        loss = self.criterion(score_view, batch.label_vec.view(-1))
        loss = loss.view(scores.shape[:-1]).sum(dim=1)
        # save loss to metrics
        notnull = batch.label_vec.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum(dim=-1)
        correct = ((batch.label_vec == preds) * notnull).sum(dim=-1)

        self.record_local_metric('loss', AverageMetric.many(loss, target_tokens))
        self.record_local_metric('ppl', PPLMetric.many(loss, target_tokens))
        self.record_local_metric(
            'token_acc', AverageMetric.many(correct, target_tokens)
        )

        # actually do backwards loss
        loss = loss.sum()
        loss /= target_tokens.sum()  # average loss per token

        if return_output:
            return (loss, model_output)
        else:
            return loss

    def eval_step(self, batch):
        """
        Evaluate a single batch of examples.
        """
        assert self.alpha >= 0

        self.model.eval()
        # 0. prime gee
        batch, gee_outputs = self._prime_gee(batch)

        # 1. Generation
        assert self.beam_size is 1
        maxlen = self.label_truncate or 256
        if not self.skip_generation:
            preds, scores = self.pragmatic_greedy_generate(batch, maxlen)
        else:
            preds = None

        # 2. Compute PPL with teacher-forced generation
        # calculate loss on targets with teacher forcing
        loss, model_output = self.compute_loss(batch, return_output=True)
        token_losses = self._construct_token_losses(
            batch.label_vec, model_output
        )

        output_texts = [self._v2t(p) for p in preds] if preds is not None else None

        # 4. update emotion posterior
        emotion_posteriors = list(torch.split(gee_outputs.emotion_posterior.cpu(), 1))

        ranked_cands = None
        return Output(output_texts, ranked_cands, token_losses=token_losses,
                      emotion_posterior=emotion_posteriors,
                      emotional_words=gee_outputs.cause_txts)

    def self_observe(self, self_message: Message) -> None:
        """
        Observe one's own utterance.

        This is used so that the agent can incorporate its own response into
        the dialogue history after a batch_act. Failure to implement this will
        result in an agent that cannot hear itself speak.

        :param self_message:
            The message corresponding to the output from batch_act.
        """
        if self_message is not None and 'emotion_posterior' in self_message:
            emotion_posterior = self_message['emotion_posterior']
            self.history._update_emotion_prior(emotion_posterior)
            del self_message['emotion_posterior']

        super().self_observe(self_message)

        return

    def _set_text_vec(self, obs, history, truncate):
        """
        Override from TorchAgent
        This will be called in super().vectorize()
        """
        super()._set_text_vec(obs, history, truncate)

        if 'prev_text_vec' not in obs:
            prev_utt_str = history.get_prev_utt_str()
            obs['prev_text'] = prev_utt_str
            if prev_utt_str:
                obs['prev_text_vec'] = history.get_prev_utt_vec()
            else:
                return obs

        # check truncation
        if obs.get('prev_text_vec') is not None:
            truncate_left = not self.history_reversed
            truncated_vec = self._check_truncate(
                obs['prev_text_vec'], truncate, truncate_left
            )
            obs.force_set('prev_text_vec', torch.LongTensor(truncated_vec))

        if 'distant_history_vec' not in obs:
            obs['distant_history_text'] = history.get_distant_history_str()
            if obs['distant_history_text']:
                obs['distant_history_vec'] = history.get_distant_history_vec()

        # check truncation
        if obs.get('distant_history_vec') is not None:
            truncate_left = not self.history_reversed
            truncated_vec = self._check_truncate(
                obs['distant_history_vec'], truncate, truncate_left
            )
            # distant_history_vec has no global_end_token yet.
            # we need to put it after when
            obs.force_set('distant_history_vec', torch.LongTensor(truncated_vec))

        # get emotion prior
        if 'emotion_prior' not in obs:
            emotion_prior = history.get_emotion_prior()
            obs['emotion_prior'] = emotion_prior

        return obs

    def _prime_gee(self, batch: Batch):
        if self.distractor_type == 'gee-focused':
            gee_results = self.gee.build_gee_distractors(batch)
        elif self.distractor_type == 'random':
            gee_results = self.gee.build_random_distractors(batch)
        else:
            raise NotImplementedError

        vectorized_world = [self._vectorize_text(d, add_start=True, add_end=True)
                               for d in gee_results.shared_world]
        world_vecs, _ = self._pad_tensor(vectorized_world)
        gee_results['world_vecs'] = world_vecs
        batch.distractor_vec = world_vecs

        return batch, gee_results

    def batchify(self, obs_batch, sort=False):
        """
        Override from TorchAgent
        Additionally batchify the distractor_text_vec and add it to batch
        """
        batch = super().batchify(obs_batch, sort)

        exs = batch.observations
        prev_xs, prev_x_lens = None, None
        if any(ex.get('prev_text') is not None for ex in exs):
            prev_xs = [ex.get('prev_text', self.EMPTY) for ex in exs]

        distant_xs, distant_xs_lens = None, None
        if any(ex.get('distant_history_text') is not None for ex in exs):
            distant_xs = [ex.get('distant_history_text', self.EMPTY) for ex in exs]

        emotion_priors = None
        if any(ex.get('emotion_prior') is not None for ex in exs):
            _emotion_priors = [ex.get('emotion_prior') for ex in exs]
            emotion_priors = torch.cat(_emotion_priors).contiguous().to(self.opt['gpu'])

        return Batch(
                prev_text=prev_xs,
                distant_history_text_list=distant_xs,
                emotion_priors=emotion_priors,
                **batch)

    def share(self):
        shared = super().share()
        shared['gee'] = self.gee

        return shared
