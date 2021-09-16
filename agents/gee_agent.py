"""
Agent for recognizing emotion cause words using GEE model
"""
import os
import torch
import torch.nn.functional as F

from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.metrics import AverageMetric
from parlai.core.torch_agent import Output
from parlai.core.torch_classifier_agent import (
    TorchClassifierAgent, ConfusionMatrixMetric, WeightedF1Metric
)
from parlai.utils.io import PathManager
from parlai.utils.typing import TShared
from parlai.utils.misc import round_sigfigs, warn_once
from parlai.core.torch_agent import Batch
from parlai.agents.bart.bart import BartAgent as OriginalBartAgent

from modules.gee_module import GeeModel
from utils.etc_utils import set_random_seed, EMOTION_LABELS
from utils.metrics import LenientF1Metric
from zoo.gee.build import download

class GeeCauseInferenceAgent(OriginalBartAgent):
    def __init__(self, opt: Opt, shared: TShared = None):
        set_random_seed(opt.get('random_seed', 15))

        # download pretrained GEE model
        download(opt['datapath'])
        self.pretrained_path = PathManager.get_local_path(
            os.path.join(opt['datapath'], 'models', 'gee', 'model')
        )
        opt['gee_checkpoint'] = self.pretrained_path
        opt['model_file'] = opt['gee_checkpoint']
        opt['classes'] = opt['emotion_classes']

        # set up classes
        if opt.get('classes') is None and opt.get('classes_from_file') is None:
            raise RuntimeError(
                'Must specify --classes or --classes-from-file argument.'
            )
        if not shared:
            if opt['classes_from_file'] is not None:
                with PathManager.open(opt['classes_from_file']) as f:
                    self.class_list = f.read().splitlines()
            else:
                self.class_list = opt['classes']
            self.class_dict = {val: i for i, val in enumerate(self.class_list)}
            if opt.get('class_weights', None) is not None:
                self.class_weights = opt['class_weights']
            else:
                self.class_weights = [1.0 for c in self.class_list]
        else:
            self.class_list = shared['class_list']
            self.class_dict = shared['class_dict']
            self.class_weights = shared['class_weights']

        super().__init__(opt, shared)

        if not shared:
            self.reset_metrics()

        # in binary classfication, opt['threshold'] applies to ref class
        if opt['ref_class'] is None or opt['ref_class'] not in self.class_dict:
            self.ref_class = self.class_list[0]
        else:
            self.ref_class = opt['ref_class']
            ref_class_id = self.class_list.index(self.ref_class)
            if ref_class_id != 0:
                # move to the front of the class list
                self.class_list.insert(0, self.class_list.pop(ref_class_id))

        # set up threshold, only used in binary classification
        if len(self.class_list) == 2 and opt.get('threshold', 0.5) != 0.5:
            self.threshold = opt['threshold']
        else:
            self.threshold = None

    def build_model(self):
        """
        Build and return model.
        """
        model = GeeModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model

    @staticmethod
    def add_cmdline_args(argparser: ParlaiParser):
        """
        Add CLI args.
        """
        TorchClassifierAgent.add_cmdline_args(argparser)
        OriginalBartAgent.add_cmdline_args(argparser)
        parser = argparser.add_argument_group('GEE Cause Inference Arguments')
        parser.add_argument(
            '--emotion-classes',
            type=str,
            nargs='*',
            default=EMOTION_LABELS,
            help='Emotion class labels'
        )
        parser.add_argument(
            '--gee-checkpoint',
            type=str,
            default='./data/models/gee/model',
        )
        parser.add_argument(
            '--topk-causes',
            type=int,
            default=-1,
            help='Specify number of causes to use'
        )

    def _set_text_vec(self, obs, history, truncate):
        obs = super()._set_text_vec(obs, history, truncate)
        if 'text_vec' in obs and 'prev_text_vec' not in obs:
            obs['prev_text'] = obs['text']
            obs['prev_text_vec'] = obs['text_vec'][1:-1]

        return obs

    def batchify(self, obs_batch, sort=False):
        batch = super().batchify(obs_batch, sort)

        exs = batch.observations

        prev_txt = None
        if any(ex.get('prev_text') is not None for ex in exs):
            prev_txt = [ex.get('prev_text', self.EMPTY) for ex in exs]

        return Batch(
            prev_text=prev_txt,
            **batch
        )

    def score(self, batch):
        """
        Define score function for classification using generative model
        """

        # vectorize emotions
        bsz = len(batch.text_vec)
        original_text_vec = batch.text_vec
        original_text_lengths = batch.text_lengths
        emotion_labels = EMOTION_LABELS
        BOS_token = torch.ones(1, dtype=torch.int64)
        EOS_token = torch.ones(1, dtype=torch.int64) * 2
        vectorized_emotions = [
            self._vectorize_text(emotion_label) for emotion_label in emotion_labels
        ]
        vectorized_emotions = [torch.cat([BOS_token, cand, EOS_token]) for cand in vectorized_emotions]
        emotion_input_vecs, emotion_vec_lengths = self._pad_tensor(vectorized_emotions)

        # set vectorized emotions as input for encoder
        batch.text_vec = emotion_input_vecs
        batch.text_lengths = emotion_vec_lengths
        emotion_cardinality = len(emotion_labels)

        # encoder inference
        encoder_states = self.model.encoder(*self._encoder_input(batch))

        # rank emotions
        ranked_emotions = []
        emotion_score_results = []
        for i in range(bsz):
            situation = original_text_vec[i, 1:]  # exclude BOS token
            tiled_situation = situation.repeat(emotion_cardinality, 1)
            scores, _ = self.model.decode_forced(encoder_states, tiled_situation)

            score_view = scores.view(-1, scores.size(-1))
            BOS_tokens = torch.ones(emotion_cardinality, 1, dtype=torch.int64).to(tiled_situation.device)
            bart_situation = torch.cat([BOS_tokens, tiled_situation], dim=1)  # bart adds BOS to input. see bart.modules
            situation_view = bart_situation.view(-1)

            losses = F.cross_entropy(
                score_view, situation_view,
                reduction='none'
            ).view(emotion_cardinality, bart_situation.size(1))

            mask = (bart_situation != self.NULL_IDX)
            mask = mask.half() if self.fp16 else mask.float()
            emotion_scores = (-losses * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
            _, ordering = emotion_scores.sort(descending=True)
            ranked_emotions.append([emotion_labels[o] for o in ordering])
            emotion_score_results.append(emotion_scores.unsqueeze(0))

        # batchify emotion scores
        batch_emotion_scores = torch.cat(emotion_score_results, dim=0).contiguous()
        return ranked_emotions, batch_emotion_scores

    def gee_cause_inference(self, batch):
        gee_outputs = self.model.reason_emotion_causes(batch)

        pred_emotions = []
        for ranked_emotion in gee_outputs['ranked_emotion']:
            pred_emotions.append(ranked_emotion[0])
        self.gee_pred_emotions = pred_emotions
        cause_texts = gee_outputs['cause_txts']

        return cause_texts

    def eval_step(self, batch):
        """
        Evaluate a single batch of examples.
        """
        if batch.text_vec is None:
            return

        self.model.eval()
        cause_texts = self.gee_cause_inference(batch)

        for topk in [1, 3, 5]:
            prec, recall, f1 = self.compute_prec_recall_f1(cause_texts, batch.observations, topk)
            self.record_local_metric(f'top{topk}_cause_recall', AverageMetric.many(recall))

        # Emotion evaluation
        ranked_emotions, scores = self.score(batch) # WARNING: this function will make change to the batch
        probs = F.softmax(scores, dim=1)
        _, prediction_id = torch.max(probs.cpu(), 1)
        preds = [self.class_list[idx] for idx in prediction_id]
        self._update_confusion_matrix(batch, preds)

        # Append keywords to preds
        preds_and_causes = []
        for pred, causes in zip(preds, cause_texts):
            new_text = 'Predicted emotion: ' + pred + ' / Predicted causes: ' + ' '.join(causes)
            preds_and_causes.append(new_text)

        if self.opt.get('print_scores', False):
            return Output(preds_and_causes, class_list=[self.class_list], probs=probs.cpu())
        else:
            return Output(preds_and_causes)

    def compute_prec_recall_f1(self, cause_texts, observations, topk):
        precs = []
        recalls = []
        f1s = []
        for pred_cause_texts, observation in zip(cause_texts, observations):
            eval_labels = [t[0] for t in observation['emotion_cause_words']]

            prec, recall, f1 = LenientF1Metric._prec_recall_f1_score(pred_cause_texts[:topk], eval_labels)
            precs.append(prec)
            recalls.append(recall)
            f1s.append(f1)

        return precs, recalls, f1s

    def share(self):
        """
        Share model parameters.
        """
        shared = super().share()
        shared['class_dict'] = self.class_dict
        shared['class_list'] = self.class_list
        shared['class_weights'] = self.class_weights
        shared['model'] = self.model
        if hasattr(self, 'optimizer'):
            shared['optimizer'] = self.optimizer
        return shared

    def _get_labels(self, batch):
        """
        Obtain the correct labels.

        Raises a ``KeyError`` if one of the labels is not in the class list.
        """
        try:
            labels_indices_list = [self.class_dict[obs['emotion']] for obs in batch.observations]
        except KeyError as e:
            warn_once('One of your labels is not in the class list.')
            raise e

        labels_tensor = torch.LongTensor(labels_indices_list)
        if self.use_cuda:
            labels_tensor = labels_tensor.cuda()
        return labels_tensor

    def _update_confusion_matrix(self, batch, predictions):
        """
        Update the confusion matrix given the batch and predictions.

        :param predictions:
            (list of string of length batchsize) label predicted by the
            classifier
        :param batch:
            a Batch object (defined in torch_agent.py)
        """
        f1_dict = {}
        for class_name in self.class_list:
            labels = [obs['emotion'] for obs in batch.observations]
            precision, recall, f1 = ConfusionMatrixMetric.compute_metrics(
                predictions, labels, class_name
            )
            f1_dict[class_name] = f1
        self.record_local_metric('emotion_classification_weighted_f1', WeightedF1Metric.compute_many(f1_dict))

    def _format_interactive_output(self, probs, prediction_id):
        """
        Format interactive mode output with scores.
        """
        preds = []
        for i, pred_id in enumerate(prediction_id.tolist()):
            prob = round_sigfigs(probs[i][pred_id], 4)
            preds.append(
                'Predicted class: {}\nwith probability: {}'.format(
                    self.class_list[pred_id], prob
                )
            )
        return preds
