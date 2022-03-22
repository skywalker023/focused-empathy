import os
import json
import spacy

from parlai.core.teachers import FixedDialogTeacher
from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data

from utils.etc_utils import EMOTION_LABELS

RESOURCES = [
    DownloadableFile(
        'https://drive.google.com/uc?id=1LR4B47Fna_l63G1X4DZtuttG-GrinnaY&export=download&confirm=t',
        'emocause.zip',
        'f490361039a98ae13a028ae3c7117ae94165fdbf771eb75d6026b63ec9d12a11',
        zipped=True, from_google=False,
    ),
]
nlp = spacy.load("en_core_web_sm")

def _build(opt):
    dpath = os.path.join(opt['datapath'], 'emocause')
    version = '1.0'

    if not build_data.built(dpath, version_string=version):
        print('[Downloading and building EmoCause data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)

    return dpath


class EmoCauseTeacher(FixedDialogTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt
        base_datatype = self.datatype.split(':')[0]
        self.datapath = _build(opt)

        if shared:
            self.data = shared['data']
        else:
            self._setup_data(base_datatype)

        # for hits@1,3,5,10
        self.metrics.eval_pr = [1, 3, 5, 10]

        self.emotion_label_set = EMOTION_LABELS

        self.num_exs = sum([len(d) for d in self.data])
        self.num_eps = len(self.data)
        self.reset()

    @classmethod
    def add_cmdline_args(cls, argparser):
        agent = argparser.add_argument_group('EmoCause teacher arguments')
        # agent.add_argument(
        #     '--add-any-argument',
        #     type=int,
        #     default=3,
        # )

    def num_episodes(self):
        return self.num_eps

    def num_examples(self):
        return self.num_exs

    def _setup_data(self, base_datatype):
        if base_datatype == 'train':
            self.data = self._setup_train_data()
            return

        episodes = []
        fname = os.path.join(
            self.datapath,
            f"{base_datatype}.json"
        )
        with open(fname, 'r') as fp:
            episodes = json.load(fp)

        new_episodes = []
        for turn in episodes:
            turn['emotion_cause_words'] = turn['annotation']
            turn['labels'] = [' '.join(turn['labels'])]
            new_episodes.append([turn])

        self.data = new_episodes

    def _setup_train_data(self):
        """
        Set up your own training data!
        """
        raise NotImplementedError

    def get(self, episode_idx, entry_idx=0):
        ep = self.data[episode_idx]
        ep_i = ep[entry_idx]

        if self.datatype == 'train':
            # TODO: set your own training scheme!
            raise NotImplementedError
        else:
            text = ep_i['original_situation']
            labels = ep_i['labels']

        action = {
            'text': text,
            'tokenized_text': ep_i['tokenized_situation'],
            'labels': labels,
            'label_candidates': self.emotion_label_set,
            'emotion': ep_i['emotion'],
            'emotion_cause_words': ep_i['emotion_cause_words'],
            'episode_done': True,
        }
        return action

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared

class DefaultTeacher(EmoCauseTeacher):
    pass
