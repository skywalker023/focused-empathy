#!/usr/bin/env python3

import torch
import torch.cuda
import torch.nn as nn

from parlai.agents.transformer.modules import TransformerGeneratorModel


class PragmaticTransformerModel(TransformerGeneratorModel):
    """
    Implements a full transformer generator model, with pragmatics
    """

    def __init__(self, opt, dictionary):
        super().__init__(opt, dictionary)

        self.alpha = 0.0 if opt.get('pragmatic_target') == 'none' else opt.get('alpha')
        self.beta = opt.get('beta')
        self.world_cardinality = opt.get('world_cardinality')
        self.worldprior = opt.get('worldprior')
        self.target_persona = 0
        self.fp16 = opt.get('fp16')

    def _initialize_worldpriors(self, bsz, seqlen):
        """
        initialize the world prior with a uniform distribution
        """
        cardinality = self.world_cardinality
        torch_dtype=torch.half if self.fp16 else torch.float
        ones = torch.ones(1, seqlen, cardinality, dtype=torch_dtype, requires_grad=False).cuda()
        uniform_world_prior = torch.log(ones / cardinality)
        world_priors = uniform_world_prior.repeat(bsz, 1, 1).detach()

        return world_priors

    def _get_initial_decoder_input(
        self, bsz: int, beam_size: int, dev: torch.device
    ) -> torch.LongTensor:
        """
        Override to seed decoder with EOS BOS token.
        """
        return (
            torch.LongTensor([self.START_IDX])
            .expand(bsz * beam_size, 1)
            .to(dev)
        )

    def _get_next_decoder_input(
        self,
        prev_input: torch.LongTensor,
        selection: torch.LongTensor,
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
        # prev_input = torch.index_select(prev_input, 0, incr_state_inds)
        decoder_input = torch.cat([prev_input, selection], dim=-1)
        return decoder_input

    def _pragmatic_reasoning(self, s0_t, worldprior):
        """
        run pragmatic reasoning with the base speaker and listener
        """

        vocab_size = self.embeddings.num_embeddings

        # log-scale
        log_score = nn.functional.log_softmax(s0_t, dim=2)
        log_score = log_score.squeeze()  # (bpsz, vocab)

        # (bsz, world_cardinality, vocab)
        log_score = log_score.view(-1, self.world_cardinality, vocab_size)

        # S_0 for L_1
        _literal_speaker = log_score.clone()
        _literal_speaker, _literal_s_next_token_idxs = torch.max(_literal_speaker, dim=-1, keepdim=True)

        # S_0 for the actual given persona (bsz, vocab)
        speaker_prior = log_score.select(1, self.target_persona)  # target persona is always index 0

        # S_0 for L_0
        # (bsz, vocab, world_cardinality)
        log_score = log_score.transpose(dim0=1, dim1=2).contiguous()
        log_score = log_score * self.beta

        # L_0 \propto S_0 * p(i)
        # worldprior should be broadcasted to all the tokens
        # (bsz, vocab, world_cardinality)
        listener_posterior = (log_score + worldprior) - torch.logsumexp(log_score + worldprior, 2, keepdim=True)

        # (bsz, vocab)
        listener_score = listener_posterior.select(2, self.target_persona)  # target persona is always index 0
        listener_score = listener_score * self.alpha

        speaker_posterior = (listener_score + speaker_prior) - torch.logsumexp(listener_score + speaker_prior, 1, keepdim=True)

        # need to unsqueeze in the dimension 1
        speaker_posterior = speaker_posterior.unsqueeze(1)  # (bsz, 1, vocab)

        # L_0 for L_1
        _literal_listener = listener_posterior.transpose(dim0=1, dim1=2).contiguous()
        _literal_listener = torch.gather(_literal_listener, -1, _literal_s_next_token_idxs)

        pragmatic_listener = (_literal_speaker + _literal_listener) - torch.logsumexp(_literal_speaker + _literal_listener, 1, keepdim=True)
        pragmatic_listener = pragmatic_listener.squeeze()

        return speaker_posterior, listener_posterior, pragmatic_listener

    def pragmatic_decode(self, encoder_states, maxlen):
        """
        greedy decoding with pragmatics
        """
        bpsz = encoder_states[0].size(0)
        bsz = bpsz // self.world_cardinality
        device = encoder_states[0].device

        # repeat encoder outputs and decoder inputs
        inputs_t = self._get_initial_decoder_input(bpsz, 1, device)

        worldpriors = self._initialize_worldpriors(bsz, maxlen).detach()

        s1_scores = []
        incr_state = None

        for t in range(maxlen):
            worldprior_t = worldpriors.select(1, t).unsqueeze(1)

            latent, incr_state = self.decoder(inputs_t, encoder_states, incr_state)
            _logits = self.output(latent)
            # only get the last timestep's logit
            s0_t = _logits.select(dim=1, index=-1).unsqueeze(1)  # logits shape: (bpsz, 1, vocab)

            # s1_t: (bsz, 1, vocab)
            # listener_posterior: (bsz, vocab, world_cardinality)
            s1_t, l0_t, l1_t = self._pragmatic_reasoning(s0_t, worldprior_t)
            s1_scores.append(s1_t)

            next_token = s1_t.max(2)[1].clone().detach()  # next input is current predicted output idx

            idx_for_tile = torch.arange(bsz).repeat(self.world_cardinality, 1).transpose(0, 1).reshape(-1).cuda()
            inputs_next_t = torch.index_select(next_token, 0, idx_for_tile)

            # update world prior
            next_token = next_token.unsqueeze(2)
            tiled_next_token = next_token.repeat(1, 1, self.world_cardinality)

            if self.worldprior != 'uniform':
                # (bsz, vocab, world_cardinality) -> (bsz, 1, world_cardinality)
                updated_world_prior = torch.gather(l0_t, 1, tiled_next_token).clone().detach()
                if t + 1 < maxlen:
                    if self.worldprior == 'L0':
                        worldpriors[:, t + 1, :] = updated_world_prior.squeeze()
                    elif self.worldprior == 'L1':
                        worldpriors[:, t + 1, :] = l1_t
                    else:
                        raise NotImplementedError

            # update inputs for next timestep: cumulate inputs_t
            inputs_t = self._get_next_decoder_input(
                inputs_t, inputs_next_t
            )

        s1_scores = torch.cat(s1_scores, dim=1)  # (bsz, seqlen, vocab)
        _, preds = s1_scores.max(dim=2)

        return preds, s1_scores

    def pragmatic_decode_forced(self, encoder_states, ys):
        """
        faster teacher-forced decoding with pragmatics
        """

        bsz = ys.size(0)
        seqlen = ys.size(1)
        self.longest_label = max(self.longest_label, seqlen)
        emb_size = self.encoder.embedding_size
        enc_outputs = encoder_states[0].view(bsz * self.world_cardinality, -1, emb_size).contiguous()
        enc_outputs_mask = encoder_states[1].view(bsz * self.world_cardinality, -1).contiguous()
        enc_states = (enc_outputs, enc_outputs_mask)
        bpsz = enc_outputs.size(0)

        # tile ys as much as the world_cardinality
        idx_for_tile = torch.arange(bsz).repeat(self.world_cardinality, 1).transpose(0, 1).reshape(-1).cuda()
        tiled_ys = torch.index_select(ys, 0, idx_for_tile)

        inputs = tiled_ys.narrow(1, 0, seqlen - 1)
        inputs = self._get_initial_forced_decoder_input(bpsz, inputs)
        self.longest_label = max(self.longest_label, seqlen)

        worldpriors = self._initialize_worldpriors(bsz, seqlen).detach()
        s1_scores = []

        latent, _ = self.decoder(inputs, enc_states)
        base_speaker = self.output(latent)

        for t in range(seqlen):

            s0_t = base_speaker.select(dim=1, index=t).unsqueeze(1)  # s0_t: (bpsz, 1, vocab)
            worldprior_t = worldpriors.select(dim=1, index=t).unsqueeze(1)

            # s1_t: (bsz, 1, vocab)
            # l0_t: (bsz, vocab, world_cardinality)
            s1_t, l0_t, l1_t = self._pragmatic_reasoning(s0_t, worldprior_t)
            s1_scores.append(s1_t)

            # Update world_prior with listener posterior
            if t + 1 < seqlen:
                next_tokens = inputs.select(1, t + 1).view(-1, 1)  # (bpsz, 1): the next tokens for each bpsz instance
                next_tokens = next_tokens.unsqueeze(2)
                # [0, 1*world_cardinality, 2*wc, 3*wc, ..., bpsz - 1wc] -> to get the ground-truth personas
                target_persona_idxs = torch.arange(bsz).cuda() * (self.world_cardinality)

                # we only need the next token of the ground-truth persona
                next_token = torch.index_select(next_tokens, 0, target_persona_idxs)  # (bsz, 1, 1)
                tiled_next_token = next_token.repeat(1, 1, self.world_cardinality)  # (bsz, 1, world_cardinality)

                if self.worldprior != 'uniform':
                    # (bsz, vocab, world_cardinality) -> (bsz, 1, world_cardinality)
                    updated_world_prior = torch.gather(l0_t, 1, tiled_next_token).clone().detach()
                    if self.worldprior == 'L0':
                        worldpriors[:, t + 1, :] = updated_world_prior.squeeze()
                    elif self.worldprior == 'L1':
                        worldpriors[:, t + 1, :] = l1_t
                    else:
                        raise NotImplementedError

        s1_scores = torch.cat(s1_scores, 1)  # (bsz, seqlen, vocab)
        _, preds = s1_scores.max(dim=2)

        return s1_scores, preds
