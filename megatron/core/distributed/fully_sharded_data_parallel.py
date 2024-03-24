import torch

import torch.distributed as dist
from . import fsdp
from .fsdp import FullyShardedDataParallel as torchFSDP
from ccdl import comm

class FullyShardedDataParallel(torchFSDP):
    def __init__(self, *args , **kwargs):
        torchFSDP.__init__(self, *args, **kwargs)
        self._warn_finish_grad_sync = False
        self._warn_zero_grad_buffer = False
        self._warn_broadcast_params = False


    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        return self.state_dict(prefix=prefix, keep_vars=keep_vars)

    def sharded_state_dict(self, prefix: str = '', sharded_offsets: torch.Tuple[torch.Tuple[int]] = ...) -> torch.Dict[str, torch.Any]:
        with torchFSDP.state_dict_type(self, fsdp.StateDictType.SHARDED_STATE_DICT):
            return self.state_dict(prefix=prefix)

    def finish_grad_sync(self):
        if not self._warn_finish_grad_sync:
            from warnings import warn
            warn("Warning: finish_grad_sync() is not supported in FSDP, and has no effect.")
            self._warn_finish_grad_sync = True

    def zero_grad_buffer(self, zero_buffer):
        if not self._warn_zero_grad_buffer:
            from warnings import warn
            warn("Warning: zero_grad_buffer() is not supported in FSDP, and has no effect.")
            self._warn_zero_grad_buffer = True

    def broadcast_params(self):
        if not self._warn_broadcast_params:
            from warnings import warn
            warn("Warning: broadcast_params() is not supported in FSDP, and has no effect.")
            self._warn_broadcast_params = True