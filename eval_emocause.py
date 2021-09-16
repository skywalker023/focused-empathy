import socket
import datetime
import os

import better_exceptions
from from_parlai.eval_model import eval_model
from from_parlai.eval_model import setup_args as eval_setupargs

better_exceptions.hook()
__PATH__ = os.path.abspath(os.path.dirname(__file__))

def setup_args(current_time):
    parser = eval_setupargs()
    parser.set_defaults(
        task='tasks.emocause',
        datapath=os.path.join(__PATH__, 'data'),
        context_length=-1,
        metrics='default',
        batchsize=8,
        display_examples=True,
        display_add_fields='emotion',
        datatype='test'
    )
    return parser

if __name__ == '__main__':
    print(f"Job is running on {socket.gethostname()}")
    current_time = datetime.datetime.now().strftime("%m%d%H%M%S")
    parser = setup_args(current_time)
    opt = parser.parse_args()
    eval_model(opt)
