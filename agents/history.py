from typing import Optional
import torch

from parlai.core.torch_agent import History
from parlai.core.message import Message


class EmpatheticHistory(History):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        opt = args[0]

        self.gpu = opt.get('gpu', 0)
        self.emotion_classes = opt.get('emotion_classes')
        self.emotion_cardinality = len(self.emotion_classes)
        self.emotion_prior = torch.log(torch.ones(1, self.emotion_cardinality) / self.emotion_cardinality)
        self.prev_utterance_strings = ''
        self.prev_utterance_vecs = []

    def reset(self):
        """Clear the history"""
        super().reset()
        self.prev_utterance_strings = ''
        self.prev_utterance_vecs = []
        self.emotion_prior = torch.log(torch.ones(1, self.emotion_cardinality) / self.emotion_cardinality)

    def _update_prev_utt_strs(self, text):
        self.prev_utterance_strings = text

    def _update_prev_utt_vecs(self, text):
        self.prev_utterance_vecs = self.parse(text)

    def _update_emotion_prior(self, posterior: torch.Tensor):
        assert self.emotion_prior.shape == posterior.shape
        self.emotion_prior = posterior

    def get_prev_utt_str(self):
        if len(self.prev_utterance_strings) == 0:
            return None
        return self.prev_utterance_strings

    def get_prev_utt_vec(self):
        if len(self.prev_utterance_vecs) == 0:
            return None
        return self.prev_utterance_vecs

    def get_emotion_prior(self):
        return self.emotion_prior

    def get_distant_history_str(self):
        """
        Return the string version of the distant history
        (i.e. dialogue history excluding the previous utterance).
        """
        if len(self.history_strings) == 1:
            return ''
        elif len(self.history_strings) > 0:
            history = self.history_strings[:-1]
            history = self.delimiter.join(history)
            if self.temp_history is not None:
                history += self.temp_history
            return history
        else:
            return None

    def get_distant_history_vec(self):
        """
        Return a vectorized version of the distant history.
        (i.e. dialogue history excluding the previous utterance).
        (dialogue history = distant history + previous utterance)
        """
        if len(self.history_vecs) == 0:
            return None

        if len(self.history_vecs) == 1:
            return []

        # vec type is a list
        history = []
        for vec in self.history_vecs[:-1]:
            history += [vec]
            history += [self.delimiter_tok]

        if self.temp_history is not None:
            history.extend([self.parse(self.temp_history)])

        history = sum(history, [])
        if self.reversed:
            history = list(reversed(history))

        return history

    def update_history(self, obs: Message, temp_history: Optional[str] = None):
        """
        Update the history with the given observation.

        :param obs:
            Observation used to update the history.
        :param temp_history:
            Optional temporary string. If it is not None, this string will be
            appended to the end of the history. It will not be in the history
            on the next dialogue turn. Set to None to stop adding to the
            history.
        """
        super().update_history(obs, temp_history=temp_history)

        if self.field in obs and obs[self.field] is not None:
            text = obs[self.field]
            self._update_prev_utt_strs(text)
            self._update_prev_utt_vecs(text)
