"""
GEE Module.
"""
import nltk
import torch
import string
import numpy as np
import torch.nn.functional as F
import pprint
from copy import deepcopy

from random import randint
from typing import Any, Dict, Union, List, Tuple
from itertools import chain

from parlai.core.opt import Opt
from parlai.core.message import Message
from parlai.core.dict import DictionaryAgent
from parlai.core.torch_agent import Batch
from parlai.core.torch_generator_agent import (
    GreedySearch, NucleusSampling, TopKSampling, BeamSearch, SearchBlocklist
)
from parlai.agents.bart.modules import BartModel
from parlai.utils.io import PathManager
from parlai.utils.misc import recursive_getattr, AttrDict
from parlai.utils.logging import logging
from parlai.utils.torch import padded_tensor


class GeeModel(BartModel):
    def __init__(self, opt: Opt, dictionary: DictionaryAgent, device=0):

        super().__init__(opt, dictionary)
        self.model_path = opt.get('gee_checkpoint')

        self.opt = opt
        self.dict = dictionary
        self.emotion_classes = opt.get('emotion_classes')
        self.emotion_cardinality = len(self.emotion_classes)
        self.update_emotion_prior = opt.get('update_emotion_prior', False)
        self.topk_causes = opt.get('topk_causes', 5)
        self.world_cardinality = opt.get('world_cardinality', 3)
        text_truncate = opt.get('text_truncate') or opt['truncate']
        self.text_truncate = text_truncate if text_truncate >= 0 else None

        self.gee_sampling = opt.get('gee-sampling', 'nucleus')

        self.beam_size = opt.get('beam_size', 1)
        self.beam_min_length = opt.get('beam_min_length', 1)
        self.beam_block_ngram = opt.get('beam_block_ngram', -1)
        self.beam_context_block_ngram = opt.get('beam_context_block_ngram', -1)
        self.beam_block_full_context = opt.get('beam_block_full_context', False)

        # load the block_list for beam search
        self.beam_block_list = self._load_beam_block_list()

        self.keyword_drop_rate = opt.get('keyword_dropout_rate', 1.0)

        self.use_cuda = not self.opt['no_cuda'] and torch.cuda.is_available()
        self.cuda_device = device
        self._build_filter_words()
        self.load(self.model_path)

    def _perspective_taking(self, batch: Batch) -> AttrDict:
        """
        First-person Perspective Taking on the context
        """
        assert hasattr(batch, 'prev_text')

        _text_vec = [torch.LongTensor(self.dict.txt2vec(t)) for t in batch.prev_text]
        text_vec, text_vec_lens = self._pad_tensor(_text_vec, device=self.cuda_device)

        bsz = len(text_vec)

        vectorized_emotions = [self._vectorize_text(e, add_start=True, add_end=True)
                               for e in self.emotion_classes]
        emotion_vec, _ = self._pad_tensor(vectorized_emotions, device=self.cuda_device)

        encoder_states = self.encoder(emotion_vec)
        tiled_encoder_states = (encoder_states[0].repeat(bsz, 1, 1),
                                encoder_states[1].repeat(bsz, 1))
        tiled_text_vec = torch.repeat_interleave(text_vec, self.emotion_cardinality, dim=0)

        # raw logits, scores also have BOS tokens so they are 1 token longer than text_vec
        scores, _ = self.decode_forced(tiled_encoder_states, tiled_text_vec)

        score_view = scores.view(-1, scores.size(-1))
        BOS_tokens = torch.ones(self.emotion_cardinality * bsz, 1, dtype=torch.int64).to(tiled_text_vec.device)
        bart_text = torch.cat([BOS_tokens, tiled_text_vec], dim=1)  # bart adds BOS to input. see bart.modules
        context_view = bart_text.view(-1)

        losses = F.cross_entropy(
            score_view, context_view,
            reduction='none'
        ).view(bsz * self.emotion_cardinality, bart_text.size(1))

        mask = (bart_text != self.NULL_IDX)
        # mask = mask.half() if self.fp16 else mask.float()
        mask = mask.float()

        # log likelihoods
        likelihoods = (-losses * mask)[:, 1:]  # remove BOS
        mask = mask[:, 1:]  # remove BOS
        text_likelihood = likelihoods.sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        text_likelihood = text_likelihood.view(bsz, self.emotion_cardinality)

        return AttrDict(text_vec=text_vec,
                        text_vec_list=_text_vec,
                        step_likelihoods=likelihoods,
                        text_likelihoods=text_likelihood,
                        losses=losses,
                        seqmask=mask)

    def recognize_emotion(self, batch: Batch, emotion_prior=None) -> AttrDict:
        perspectives = self._perspective_taking(batch)

        text_likelihood = perspectives.text_likelihoods
        bsz = len(text_likelihood)

        # init uniform prior
        if emotion_prior is None:
            ones = torch.ones(bsz, self.emotion_cardinality).to(text_likelihood.device)
            emotion_prior = torch.log(ones / self.emotion_cardinality)
        else:
            assert isinstance(emotion_prior, torch.Tensor) and emotion_prior.shape == text_likelihood.shape

        emotion_posterior = text_likelihood + emotion_prior
        _, orderings = emotion_posterior.sort(descending=True)

        ranked_emotions = []
        for order in orderings:
            ranked_emotions.append([self.emotion_classes[o] for o in order])

        return AttrDict(emotion_posterior=emotion_posterior,
                        ranked_emotion=ranked_emotions,
                        ranked_emotion_idx=orderings,
                        **perspectives)

    def reason_emotion_causes(self, batch: Batch, emotion_prior=None, target_topk_emotions=2) -> AttrDict:
        """
        Recognize emotion cause words in Bayesian fashion
        """

        perspectives = self.recognize_emotion(batch, emotion_prior)

        bsz = len(perspectives.text_vec)
        SCORE_MASK = -1000

        token_likelihoods = perspectives.step_likelihoods.view(bsz, self.emotion_cardinality, -1)
        seqlen = token_likelihoods.shape[-1]
        marginal_emo = perspectives.emotion_posterior.unsqueeze(-1)

        # p(u_t | E, u_<t) * p(E)
        token_emo_jointprob = token_likelihoods + marginal_emo

        def _compute_token_topkemo_jointprob(perspectives, target_topk=2):
            """
            compute p(w_t, topk_emotion | w_<t)
            """
            text_vec = perspectives.text_vec
            bsz = len(text_vec)

            recognized_emotions = [e[:target_topk] for e in perspectives.ranked_emotion]
            _vectorized_emotions = [torch.cat(list(map(self._vectorize_text, e)), dim=0)
                                    for e in recognized_emotions]
            vectorized_emotions = [self._vectorize_text(e, add_start=True, add_end=True, truncate=self.text_truncate)
                                for e in _vectorized_emotions]
            emotion_vec, _ = self._pad_tensor(vectorized_emotions, device=self.cuda_device)

            encoder_states = self.encoder(emotion_vec)
            scores, _ = self.decode_forced(encoder_states, text_vec)

            score_view = scores.view(-1, scores.size(-1))
            BOS_tokens = torch.ones(bsz, 1, dtype=torch.int64).to(text_vec.device)
            bart_text = torch.cat([BOS_tokens, text_vec], dim=1)  # bart adds BOS to input. see bart.modules
            context_view = bart_text.view(-1)

            losses = F.cross_entropy(
                score_view, context_view,
                reduction='none'
            ).view(bsz, bart_text.size(1))

            mask = (bart_text != self.NULL_IDX)
            # mask = mask.half() if self.fp16 else mask.float()
            mask = mask.float()

            # log likelihoods
            topk_emo_token_likelihoods = (-losses * mask)[:, 1:].unsqueeze(1)  # remove BOS
            topk_emo_idx = perspectives.ranked_emotion_idx[:, :target_topk]
            topk_emo_marginals = torch.gather(perspectives.emotion_posterior, 1, topk_emo_idx)
            marginal_topk_emo = torch.logsumexp(topk_emo_marginals, dim=1, keepdim=True).unsqueeze(-1)

            # p(w_t | e, w_<t) * p(e)
            token_topkemo_joint = topk_emo_token_likelihoods + marginal_topk_emo  

            return token_topkemo_joint

        # compute p(w_t, topk_emotion | w_<t)
        token_topkemotion_jointprob = _compute_token_topkemo_jointprob(perspectives, target_topk_emotions)  

        # collect the top1, bottom2, bottom1 emotions
        gather_rank_idxs = torch.tensor([[0, self.emotion_cardinality - 2, self.emotion_cardinality - 1]],
                                        dtype=torch.int64).repeat(bsz, 1).to(perspectives.text_vec.device)
        marginal_emo_idxs = torch.gather(perspectives.ranked_emotion_idx, dim=1, index=gather_rank_idxs)
        marginal_emo_idxs = marginal_emo_idxs.unsqueeze(-1).repeat(1, 1, seqlen)

        # collect the jointprob of top1, bottom2, bottom1 emotions for approximating P(e|w_t)
        token_emo_jointprob_subset = torch.gather(token_emo_jointprob, dim=1, index=marginal_emo_idxs) 
        token_emo_jointprob_subset[:, 0, :] = token_topkemotion_jointprob.squeeze()  # overwrite top1 with p(w_t, topk_e | w_<t)

        # p(e | w_t) = p(w_t, e | w_<t) / p(w_t | w_<t)
        token_emos = token_emo_jointprob_subset - torch.logsumexp(token_emo_jointprob_subset, dim=1, keepdim=True)

        # p(e_top | u_t, u_<t)
        token_top1_emo = token_emos[:, 0, :]
        gather_top1_idxs = marginal_emo_idxs[:, 0, :].unsqueeze(1)
        seqlen_mask = perspectives.seqmask.view(bsz, self.emotion_cardinality, seqlen)
        _top1_emo_seqmask = torch.gather(seqlen_mask, dim=1, index=gather_top1_idxs).squeeze()
        top1_emo_seqmask = (_top1_emo_seqmask == self.NULL_IDX).float() * SCORE_MASK
        emo_cause_scores = token_top1_emo + top1_emo_seqmask
        # emo_cause_scores[:, 0] = SCORE_MASK # remove BOS scores
        emo_cause_probs = torch.softmax(emo_cause_scores, dim=1)

        # rank the most probable cause in respect to idxs and tokens
        sorted_emo_cause_probs, cause_orderings = emo_cause_probs.sort(descending=True)
        sorted_tok_idxs = torch.gather(perspectives.text_vec, dim=1, index=cause_orderings)
        sorted_cause = np.array([self._index2token(idxs) for idxs in sorted_tok_idxs])

        # filter stopwords
        filtered_outputs = self._filter_cause_tokens(sorted_cause, cause_orderings,
                                                     sorted_tok_idxs, sorted_emo_cause_probs)

        # truncate causes
        if self.topk_causes > 0:
            topk = self.topk_causes
            for idx, output in enumerate(filtered_outputs):
                filtered_outputs[idx] = [o[:topk] if len(o) > topk else o for o in output]

        # recover original words from subword causes
        cause_word_outputs = self._recover_words_from_subwords(causes=filtered_outputs[0],
                                                               cause_tok_loc=filtered_outputs[1],
                                                               original_text_vec=perspectives.text_vec)

        return AttrDict(
                **cause_word_outputs,
                cause_probs=filtered_outputs[3],  # cause probs in respect to tokens, not recovered words
                **perspectives
        )

    def build_gee_distractors(self,
                              batch: Batch,
                              priming_bottomk_emotion: int = 3,
                              emotion_prior: torch.LongTensor = None) -> AttrDict:
        """
        Build distractors by priming gee with previous utterance's cause tokens masked:
        """
        emotion_prior = batch.emotion_priors.to(self.cuda_device) if self.update_emotion_prior else None
        perspectives = self.reason_emotion_causes(batch, emotion_prior)

        text_vec_list = perspectives.text_vec_list
        cause_locs = perspectives.cause_locs
        cause_vecs = perspectives.cause_vecs
        distractor_size = self.world_cardinality - 1
        device = self.cuda_device
        distractor_list = [text_vec_list]
        SCORE_MASK = -100

        # Mask those cause_locations and prime gee to generate distractors
        masks = [torch.ones_like(vec, dtype=torch.bool) for vec in text_vec_list]
        for mask, loc in zip(masks, cause_locs):
            mask[loc] = False

        _priming_tokens = [torch.masked_select(vec, mask) for vec, mask in zip(text_vec_list, masks)]

        # vectorize and concat bottom-k emotions (i.e., the most unlikely emotions)
        recognized_emotions = [e[-priming_bottomk_emotion:] for e in perspectives.ranked_emotion]
        vectorized_emotions = [torch.cat(list(map(self._vectorize_text, e)), dim=0)
                            for e in recognized_emotions]

        # concat recognized emotion to the priming tokens
        priming_tokens = [self._vectorize_text(torch.cat([e, t]),
                                            add_start=True, add_end=True, truncate=self.text_truncate)
                                            for e, t in zip(vectorized_emotions, _priming_tokens)]

        # batchify
        priming_vec, _ = self._pad_tensor(priming_tokens, device=device)

        self.longest_label = max(self.longest_label, perspectives.text_vec.size(1))
        encoder_states = self.encoder(priming_vec)
        scores, _ = self.decode_forced(encoder_states, perspectives.text_vec)

        if scores.size(1) != perspectives.text_vec.size(1):
            scores = scores[:, 1:, :]  # ignore start

        # select topk tokens from gee as distractor tokens, excluding the original cause token
        distractor_toks = []
        special_toks = [0, 1, 2, 3, 4, 50261, 50262, 50263, 50264]
        for i, c in enumerate(cause_locs):
            s = scores[i, c].index_fill_(1, cause_vecs[i], SCORE_MASK)
            replacing_toks = torch.softmax(s / 2.0, dim=1).multinomial(distractor_size)  # sample

            # while any(pad in replacing_toks for pad in special_toks):
            for _ in range(10):
                replacing_toks = torch.softmax(s / 2.0, dim=1).multinomial(distractor_size)  # sample
                if not any(pad in replacing_toks for pad in special_toks):
                    break

            replacing_tokens = [r.squeeze() for r in replacing_toks.T.split(1, dim=0)]
            distractor_toks.append([cause_vecs[i]] + replacing_tokens)

        # replace the original cause tokens with gee-generated tokens
        distractor_list = []
        for i, text in enumerate(perspectives.text_vec_list):
            for replacement in distractor_toks[i]:
                text[cause_locs[i]] = replacement.cpu()
                distractor_list.append(deepcopy(text))

        distractor_text = [self._v2t(d) for d in distractor_list]

        shared_world = self._set_shared_world(batch.distant_history_text_list, distractor_text)

        return AttrDict(shared_world=shared_world,
                        **perspectives)

    def build_random_distractors(self, batch: Batch) -> AttrDict:

        emotion_prior = batch.emotion_priors.to(self.cuda_device) if self.update_emotion_prior else None
        perspectives = self.reason_emotion_causes(batch, emotion_prior)

        bsz = len(batch.text_vec)
        world = self.world_cardinality

        prev_texts = [batch.observations[i]['prev_text'] for i in range(bsz)]

        # do in-batch shift for distractors (i.e. simply use other previous texts in the batch as random distractors)
        distractor_text = []
        for i, t in enumerate(prev_texts):
            distractor_text.append(t)
            for n in range(i + 1, i + world):
                distractor_text.append(prev_texts[n % bsz])

        shared_world = self._set_shared_world(batch.distant_history_text_list, distractor_text)

        return AttrDict(shared_world=shared_world,
                        **perspectives)

    def _set_shared_world(self, common_history, distractor_text):
        """
        Concatenate common history with distractor_texts (including gt and distractors)
        """

        batch_history = [h for h in common_history for _ in range(self.world_cardinality)]

        shared_world = []
        for h, d in zip(batch_history, distractor_text):
            if not h:
                shared_world.append(d)
            else:
                shared_world.append('\n'.join([h, d]))

        return shared_world

    def generate_episodes(self, batch: Batch, max_ts: int):
        """
        Generate emotional situations when given an emotion.
        """
        vectorized_emotions = [self._vectorize_text(e, add_start=True, add_end=True)
                               for e in self.emotion_classes]
        emotion_vec, _ = self._pad_tensor(vectorized_emotions, device=self.cuda_device)
        batch.observations = tuple(Message(full_text_vec=v[1:-1]) for v in vectorized_emotions)

        emotional_episodes = self._sample_episodes(batch, emotion_vec, max_ts=64)

        return emotional_episodes

    def _sample_episodes(self, batch: Batch, priming_vec: torch.LongTensor, max_ts: int) -> List[torch.Tensor]:
        _episodes_list = []
        num_sample = 10
        for _ in range(num_sample):
            beam_preds_scores, beams = self._generate(batch, priming_vec=priming_vec, max_ts=max_ts)
            preds, scores = zip(*beam_preds_scores)

            # Remove BOS, EOS token from preds
            _episodes = [p[1:-1].cpu() for p in preds]
            _episodes_list.append(_episodes)

        # transpose and flatten it
        _transposed_episode_vecs = [list(d) for d in zip(*_episodes_list)]
        episode_list = list(chain(*_transposed_episode_vecs))

        _episode_texts = [self._v2t(e) for e in episode_list]
        episode_texts = [_episode_texts[i * num_sample: (i + 1) * num_sample] for i in range(len(self.emotion_classes))]
        emotional_episodes = list(zip(self.emotion_classes, episode_texts))

        return emotional_episodes

    def load(self, path: str) -> Dict[str, Any]:
        """
        Return opt and model states.

        Override this method for more specific loading.
        """
        print(f"Load {path}")
        import parlai.utils.pickle

        with PathManager.open(path, 'rb') as f:
            states = torch.load(
                f, map_location=lambda cpu, _: cpu, pickle_module=parlai.utils.pickle
            )
        if 'model' in states:
            state_dict = states['model']
            try:
                self.load_state_dict(state_dict)
            except RuntimeError as msg:
                msg_ = str(msg)
                if 'size mismatch' in msg_ and 'embedding' in msg_:
                    if hasattr(self, 'special_toks') and len(self.special_toks) > 0:
                        state_dict = self._resize_token_embeddings(state_dict, msg_)
                        self.load_state_dict(state_dict)
                        self.resized_embeddings = True  # make note that we resized here
                    else:
                        raise RuntimeError(
                            f'{msg_}\n'
                            '-----------------\n'
                            'Could not load the model due to a size mismatch in the '
                            'embeddings. A common reason for this is trying to load '
                            'a model trained with fp16 but loaded without fp16. Try '
                            'adding --fp16 true or --force-fp16-tokens true.'
                        )

        return states

    def _build_filter_words(self):
        self.filterwords = nltk.corpus.stopwords.words('english') + ['us', 'oh', 'yeah', 'but', 'bit']
        self.filterwords += list(string.punctuation)
        self.filterwords += [self.dict.null_token]
        self.filterwords += ['', '...', '."', "'m", "'s", "'t", "'ve", "'ll", "nt"]

    def _filter_cause_tokens(self, cause_toks: np.array, *args: torch.Tensor):
        """
        Assume cause tokens are sorted in desending order in respect to probability.
        Remove stopword tokens and punctuations.
        """
        shape = cause_toks.shape
        assert all(arg.shape == shape for arg in args)

        BPE_PREFIX='\\xc4\\xa0'

        # boolean index for non-stopword tokens
        bool_idxs = []
        for ct in cause_toks:
            _bools = list(map(lambda x: x.replace(BPE_PREFIX, '').lower() not in self.filterwords, ct))
            # if all causes are words to be filtered, we let the most probable one remain
            if not any(_bools):
                _bools[0] = True
            bool_idxs.append(_bools)
        bool_idxs = np.array(bool_idxs)

        # index for partitioning according to batch,
        # because dimensions are lost after boolean indexing
        batch_partition = np.cumsum(bool_idxs.sum(axis=1))[:-1]

        # extract with boolean indexing and split to batch
        targets = [cause_toks] + [arg.cpu().numpy() for arg in args]
        filtered_outputs = [np.split(target[bool_idxs], batch_partition) for target in targets]

        return filtered_outputs

    def _recover_words_from_subwords(
            self, causes: List[np.array], cause_tok_loc: List[np.array], original_text_vec: torch.Tensor
    ):
        """
        Find the remaining subword parts in a recursive fashion and recover the original word
        """
        BPE_PREFIX='\\xc4\\xa0'
        punct = string.punctuation
        original_text = [self._index2token(indices, remove_special_token=True)
                         for indices in original_text_vec]

        def find_right(loc, original_text):
            # return nothing when 
            # it reaches the end of the text
            # or the right token starts with the BPE prefix
            # or the right token is a punctuation
            if loc >= len(original_text) or original_text[loc].startswith(BPE_PREFIX) or original_text[loc] in punct:
                return []
            current = [loc]
            right = find_right(loc + 1, original_text)
            return current + right

        def find_left(loc, original_text):
            # return nothing when it reaches the start of the text
            if loc < 0:
                return []

            # if the left token has a BPE prefix, then we should add it and end search
            if original_text[loc].startswith(BPE_PREFIX):
                return [loc]

            # if the left token also does not start with BPE prefix, we add it and search further
            current = [loc]
            left = find_left(loc - 1, original_text)
            return left + current

        def merge_tokens(recovered_tokens):
            merged_tokens = []
            merged = ''
            for i, tok in recovered_tokens:
                if i == 0:
                    merged += tok
                else:
                    if tok.startswith(BPE_PREFIX):
                        merged_tokens.append(merged.replace(BPE_PREFIX, ''))
                        merged = tok
                    else:
                        merged += tok
            
            return merged_tokens

        # find the locations of the rest of the subwords
        locs_batch = []
        merged_words_batch = []
        recovered_text_vec_batch = []
        for batch_idx, (cause, locs) in enumerate(zip(causes, cause_tok_loc)):
            merged_words = []
            locs_of_tokens = []
            recovered_text_vec = []
            for token, location in zip(cause, locs):
                # skip subwords which were already used for recovering the word
                if location not in locs_of_tokens:
                    right = find_right(location + 1, original_text[batch_idx])
                    if not token.startswith(BPE_PREFIX):
                        left = find_left(location - 1, original_text[batch_idx])
                    else:
                        left = []
                    original_word_locs = left + [location] + right

                    locs_of_tokens += original_word_locs
                    merged_w = ''.join(original_text[batch_idx][original_word_locs])
                    merged_words.append(merged_w.replace(BPE_PREFIX, ''))
                    recovered_text_vec += original_text_vec[batch_idx][original_word_locs]

            locs_batch.append(np.array(locs_of_tokens))
            merged_words_batch.append(merged_words)
            recovered_text_vec_batch.append(torch.stack(recovered_text_vec))

        return AttrDict(cause_txts=merged_words_batch,
                        cause_vecs=recovered_text_vec_batch,
                        cause_locs=locs_batch)  # location of cause words in the text_vec

    def _generate(self, batch: Batch, priming_vec: torch.LongTensor, max_ts: int):
        """
        Generate emotional situations
        """
        bsz = len(priming_vec)
        beam_size = self.beam_size
        device = self.cuda_device
        observations = deepcopy(batch.observations)
        gee_batch = Batch(text_vec=priming_vec, observations=observations)
        beams = [
            self._treesearch_factory(device)
            .set_context(self._get_context(gee_batch, batch_idx))
            .set_block_list(self.beam_block_list)
            for batch_idx in range(bsz)
        ]

        encoder_states = self.encoder(priming_vec)

        # repeat encoder outputs and decoder inputs
        decoder_input = self._get_initial_decoder_input(bsz, beam_size=beam_size, dev=device)

        inds = torch.arange(bsz).to(device).unsqueeze(1).repeat(1, beam_size).view(-1)
        encoder_states = self.reorder_encoder_states(encoder_states, inds)

        incr_state = None
        for _ts in range(max_ts):
            score, incr_state = self.decoder(decoder_input, encoder_states, incr_state)
            # only need the final hidden state to make the word prediction
            score = score[:, -1:, :]
            score = self.output(score)
            # score contains softmax scores for bsz * beam_size samples
            score = score.view(bsz, beam_size, -1)
            # force to fp32 to avoid overflow issues during search calculations
            score = F.log_softmax(score, dim=-1, dtype=torch.float32)  # type: ignore

            for i, b in enumerate(beams):
                if not b.is_done():
                    b.advance(score[i])

            # update incr_state
            incr_state_inds = torch.cat(
                [
                    beam_size * i + b.get_backtrack_from_current_step()
                    for i, b in enumerate(beams)
                ]
            )
            incr_state = self.reorder_decoder_incremental_state(
                incr_state, incr_state_inds
            )
            # selection: next input for decoder_input
            selection = torch.cat(
                [b.get_output_from_current_step() for b in beams]
            ).unsqueeze(-1)
            decoder_input = self._get_next_decoder_input(
                decoder_input, selection, incr_state_inds
            )

        # get all finalized candidates for each sample (and validate them)
        n_best_beam_preds_scores = [b.get_rescored_finished() for b in beams]

        if hasattr(self, '_rerank_beams'):
            n_best_beam_preds_scores = self._rerank_beams(  # type: ignore
                batch, n_best_beam_preds_scores
            )

        # get the top prediction for each beam (i.e. minibatch sample)
        beam_preds_scores = [n_best_list[0] for n_best_list in n_best_beam_preds_scores]

        return beam_preds_scores, beams

    def _set_distractor_input(
        self, batch: Batch, distractor_batch: List[torch.Tensor],
        add_start=True, add_end=True
    ):
        assert hasattr(batch, 'distant_history_text_list')

        distant_history = [torch.LongTensor(self.dict.txt2vec(h)) for h in batch.distant_history_text_list]

        # Since, the result of this function is given as input for the main agent,
        # it must be moved to the main agent's cuda device
        gpu = self.opt.get('gpu')

        if distant_history is None:
            # distractor previous utterance is all we need
            distractor_batch = [self._vectorize_text(distractor, add_start=add_start,
                                                     add_end=add_end, truncate=self.text_truncate)
                                for distractor in distractor_batch]
        else:
            # 1. copy history tensors for world_cardinality. Should we copy or clone?
            history_batch = [h for h in distant_history for _ in range(self.world_cardinality)]
            # 2. concat with distractor_batch
            # 3. add_start_end_tokens & truncate = self._vectorize_text()
            distractor_batch = [self._vectorize_text(torch.cat([h, d]),
                                                    add_start=add_start, add_end=add_end,
                                                    truncate=self.text_truncate)
                                for h, d in zip(history_batch, distractor_batch)]
        # batchify
        distractor_vec, distractor_vec_lengths = self._pad_tensor(distractor_batch)

        return distractor_vec, distractor_vec_lengths


    def _treesearch_factory(self, device):
        method = self.gee_sampling
        beam_size = self.opt.get('beam_size', 1)
        if method == 'greedy':
            return GreedySearch(
                beam_size,
                min_length=0,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=device,
            )
        elif method == 'beam':
            return BeamSearch(
                beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=device,
            )
        elif method == 'topk':
            return TopKSampling(
                self.opt['topk'],
                beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=device,
            )
        elif method == 'nucleus':
            return NucleusSampling(
                self.opt['topp'],
                beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=device,
            )
        else:
            raise ValueError(f"Can't use inference method {method}")

    def _load_beam_block_list(self) -> SearchBlocklist:
        """
        Load the beam block_list.

        :return: a dict mapping ngram length to different ngrams
        """
        block_list = SearchBlocklist(self.dict)
        if not self.opt.get('beam_block_list_filename'):
            return block_list

        block_list_fn = self.opt['beam_block_list_filename']
        try:
            with PathManager.open(block_list_fn) as f:
                for line in f:
                    block_list.add(line.strip())
        except IOError:
            logging.error(
                f"Could not load beam block_list {block_list_fn}, using empty block_list."
            )
        return block_list

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

    def _resize_token_embeddings(self, state_dict, msg=None):
        """
        Must define this for your agent if you wish to add additional special tokens.
        Must make a call to resize the token embeddings and load the model state dict
        with the resized token embeddings.

        Resize the token embeddings when are adding extra special tokens.
        """
        new_size = self.embeddings.weight.size()[0]
        original_size = state_dict['embeddings.weight'].size()[0]
        logging.info(f'Resizing token embeddings from {original_size} to {new_size}')
        if new_size <= original_size:
            raise RuntimeError("New size should be bigger than the original size!")

        for emb_weights in [
            'embeddings.weight',
            'encoder.embeddings.weight',
            'decoder.embeddings.weight',
        ]:
            # get new_embs
            original_embs = state_dict[emb_weights]
            new_embs = recursive_getattr(self.modules, emb_weights).to(original_embs.device)

            # copy over old weights
            new_embs.data[:original_size, :] = original_embs.data[:original_size, :]

            # reset in state dict
            state_dict[emb_weights] = new_embs

        return state_dict

    def _index2token(self, idxs, remove_prefix=False, remove_special_token=False):
        """
        Convert token indices to string of tokens.
        """
        BPE_PREFIX='\\xc4\\xa0'
        tokens = []
        if hasattr(idxs, 'cpu'):
            idxs = idxs.cpu()
        for i in idxs:
            # if not i in [self.START_IDX, self.END_IDX, self.NULL_IDX]:
            tok = self.dict.ind2tok[int(i)]
            if remove_prefix and tok.startswith(BPE_PREFIX):
                tok = tok.replace(BPE_PREFIX, '')
            if remove_special_token and i in [self.START_IDX, self.END_IDX, self.NULL_IDX]:
                continue
            else:
                tokens.append(tok)

        return np.array(tokens)

    def _v2t(self, vec):
        """
        Convert token indices to string of tokens.
        """
        new_vec = []
        if hasattr(vec, 'cpu'):
            vec = vec.cpu()
        for i in vec:
            if i == self.END_IDX:
                break
            elif i != self.START_IDX:
                new_vec.append(i)
        return self.dict.vec2txt(new_vec)

    def _get_initial_decoder_input(
        self, bsz: int, beam_size: int, dev: torch.device
    ) -> torch.LongTensor:
        """
        Override to seed decoder with EOS BOS token.
        """
        return (
            torch.LongTensor(  # type: ignore
                [self.END_IDX, self.START_IDX]
            )
            .expand(bsz * beam_size, 2)
            .to(dev)
        )

    def _get_next_decoder_input(
        self,
        prev_input: torch.LongTensor,
        selection: torch.LongTensor,
        incr_state_inds: torch.LongTensor,
    ) -> torch.LongTensor:
        """
        Return next decoder input.

        :param prev_input:
            previous input to decoder
        :param selection:
            token selections for current timestep
        :param inds:
            incremental state indices

        :return decoder input:
            return decoder input for next timestep
        """
        prev_input = torch.index_select(prev_input, 0, incr_state_inds)
        decoder_input = torch.cat([prev_input, selection], dim=-1)
        return decoder_input

    def _add_start_end_tokens(self, vec, add_start=False, add_end=False):
        """
        Add start and end tokens to a list or tensor.
        """
        if isinstance(vec, torch.Tensor):
            if len(vec.shape) != 1:
                raise Exception('_add_start_end_tokens expects a 1D tensor')
            tensors = [vec]
            if add_start:
                tensors.insert(0, vec.new_tensor([self.START_IDX]))
            if add_end:
                tensors.append(vec.new_tensor([self.END_IDX]))
            return torch.cat(tensors, 0)
        if add_start:
            vec.insert(0, self.START_IDX)
        if add_end:
            vec.append(self.END_IDX)
        return vec

    def _check_truncate(self, vec, truncate, truncate_left=False):
        """
        Check that vector is truncated correctly.
        """
        if truncate is None:
            return vec
        if len(vec) <= truncate:
            return vec
        if truncate_left:
            return vec[-truncate:]
        else:
            return vec[:truncate]

    def _vectorize_text(
        self, text, add_start=False, add_end=False, truncate=None, truncate_left=True
    ):
        """
        Return vector from text.

        :param text:
            String to vectorize or vectorized text

        :param add_start:
            Add the start token to the front of the tensor.

        :param add_end:
            Add the end token to the end of the tensor.

        :param truncate:
            Truncate to this many tokens >= 0, or None.

        :param truncate_left:
            Truncate from the left side (keep the rightmost tokens). You
            probably want this True for inputs, False for targets.
        """
        if isinstance(text, str):
            vec = self.dict.txt2vec(text)
        elif isinstance(text, torch.Tensor):
            vec = text
        else:
            raise TypeError

        vec = self._add_start_end_tokens(vec, add_start, add_end)
        vec = self._check_truncate(vec, truncate, truncate_left)
        if isinstance(vec, torch.Tensor) and vec.is_cuda:
            tensor = torch.cuda.LongTensor(vec) if not isinstance(vec, torch.cuda.LongTensor) else vec
        else:
            tensor = torch.LongTensor(vec) if not isinstance(vec, torch.LongTensor) else vec
        return tensor

    def _get_context(self, batch, batch_idx):
        """
        Set the beam context for n-gram context blocking.

        Intentionally overridable for more complex model histories.
        """
        ctxt = batch.text_vec[batch_idx]
        if self.beam_block_full_context:
            full_ctxt = batch.observations[batch_idx].get('full_text_vec', ctxt)
            if not isinstance(full_ctxt, torch.Tensor):
                full_ctxt = torch.LongTensor(full_ctxt).to(ctxt.device)
            ctxt = full_ctxt
        return ctxt
