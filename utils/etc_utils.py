import os
import random
import torch
import numpy as np

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

EMOTION_LABELS = ['__afraid__', '__angry__', '__annoyed__', '__anticipating__', '__anxious__',
                  '__apprehensive__', '__ashamed__', '__caring__', '__confident__', '__content__',
                  '__devastated__', '__disappointed__', '__disgusted__', '__embarrassed__',
                  '__excited__', '__faithful__', '__furious__', '__grateful__', '__guilty__',
                  '__hopeful__', '__impressed__', '__jealous__', '__joyful__', '__lonely__',
                  '__nostalgic__', '__prepared__', '__proud__', '__sad__', '__sentimental__',
                  '__surprised__', '__terrified__', '__trusting__']
