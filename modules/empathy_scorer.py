import os
import numpy as np

import torch
import torch.nn as nn
import parlai.core.build_data as build_data
from transformers import RobertaTokenizer

from from_epitome.models import (
    BiEncoderAttentionWithRationaleClassification
)


RESOURCES = [
    build_data.DownloadableFile(
        '1P3Gd4uEzH-SS0L9K5TOktsPlR9rKU5sv',
        'finetuned_EX.pth',
        '4f43ceb2526e008a2093856208abb878f14236dd54e4fdcdfdd4ccbeb9c08178',
        zipped=False, from_google=True,
    ),
    build_data.DownloadableFile(
        '1Ta5PvUV-UFFWUa_WmyT0YFYez_XL6bb2',
        'finetuned_IP.pth',
        'e80d1bcfb75f7046961ed71cfd2eada2d939f7f1191e169b0b4aa68e9b6054dc',
        zipped=False, from_google=True,
    ),
]


def _build(datapath):
    dpath = os.path.join(datapath, 'models', 'epitome')
    version = '1.0'

    if not build_data.built(dpath, version_string=version):
        print('[Downloading and building empathy scorer: ' + dpath + ']')
        print('NOTE: The download can take about 4 minutes (likely to vary depending on your internet speed)')
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


class EmpathyScorer(nn.Module):
    def __init__(self, opt, batch_size=1, cuda_device=0):
        print("Loading EmpathyScorer...")
        super().__init__()
        ckpt_path = _build(opt['datapath'])

        self.opt = opt
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        self.batch_size = batch_size
        self.cuda_device = cuda_device

        self.model_IP = BiEncoderAttentionWithRationaleClassification()
        self.model_EX = BiEncoderAttentionWithRationaleClassification()

        IP_weights = torch.load(os.path.join(ckpt_path, 'finetuned_IP.pth'))
        self.model_IP.load_state_dict(IP_weights)

        EX_weights = torch.load(os.path.join(ckpt_path, 'finetuned_EX.pth'))
        self.model_EX.load_state_dict(EX_weights)

        self.use_cuda = not opt['no_cuda'] and torch.cuda.is_available()
        if self.use_cuda:
            self.model_IP.cuda(self.cuda_device)
            self.model_EX.cuda(self.cuda_device)

    def forward(self, seeker_post, response_post):
        self.model_IP.eval()
        self.model_EX.eval()

        # 'input_ids', 'attention_mask'
        seeker_input = self.tokenizer.batch_encode_plus(
            seeker_post,
            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
            max_length = 64,           # Pad & truncate all sentences.
            truncation=True,
            pad_to_max_length = False,
            return_attention_mask = True,   # Construct attn. masks.
            return_tensors = 'pt',     # Return pytorch tensors.
            padding=True,
        )
        response_input = self.tokenizer.batch_encode_plus(
            response_post,
            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
            max_length = 64,           # Pad & truncate all sentences.
            truncation=True,
            pad_to_max_length = False,
            return_attention_mask = True,   # Construct attn. masks.
            return_tensors = 'pt',     # Return pytorch tensors.
            padding=True
        )
        if self.use_cuda:
            # We assume all parameters of the model are on the same cuda device
            device = next(self.model_IP.parameters()).device
            seeker_input['input_ids'] = seeker_input['input_ids'].to(device)
            seeker_input['attention_mask'] = seeker_input['attention_mask'].to(device)
            response_input['input_ids'] = response_input['input_ids'].to(device)
            response_input['attention_mask'] = response_input['attention_mask'].to(device)
        with torch.no_grad():
            (logits_empathy_IP, logits_rationale_IP,) = self.model_IP(
                input_ids_SP=seeker_input['input_ids'],
                input_ids_RP=response_input['input_ids'],
                token_type_ids_SP=None,
                token_type_ids_RP=None,
                attention_mask_SP=seeker_input['attention_mask'],
                attention_mask_RP=response_input['attention_mask']
            )
            (logits_empathy_EX, logits_rationale_EX,) = self.model_EX(
                input_ids_SP=seeker_input['input_ids'],
                input_ids_RP=response_input['input_ids'],
                token_type_ids_SP=None,
                token_type_ids_RP=None,
                attention_mask_SP=seeker_input['attention_mask'],
                attention_mask_RP=response_input['attention_mask']
            )

        logits_empathy_IP = logits_empathy_IP.detach().cpu().numpy()
        logits_rationale_IP = logits_rationale_IP.detach().cpu().numpy()
        empathy_predictions_IP = np.argmax(logits_empathy_IP, axis=1).tolist()
        rationale_predictions_IP = np.argmax(logits_rationale_IP, axis=2)

        logits_empathy_EX = logits_empathy_EX.detach().cpu().numpy()
        logits_rationale_EX = logits_rationale_EX.detach().cpu().numpy()
        empathy_predictions_EX = np.argmax(logits_empathy_EX, axis=1).tolist()
        rationale_predictions_EX = np.argmax(logits_rationale_EX, axis=2)

        return {'IP': (empathy_predictions_IP, rationale_predictions_IP),
                'EX': (empathy_predictions_EX, rationale_predictions_EX)}
