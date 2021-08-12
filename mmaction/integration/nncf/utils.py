import torch
from collections import OrderedDict
from contextlib import contextmanager

try:
    import nncf
    _is_nncf_enabled = True
except ImportError:
    _is_nncf_enabled = False
except RuntimeError as _e:
    _is_nncf_enabled = False
    print('Attention: RuntimeError happened when tried to import nncf')
    print('           The reason may be in absent CUDA devices')
    print('           RuntimeError:')
    print('           ' + str(_e), flush=True)



def is_nncf_enabled():
    return _is_nncf_enabled


def check_nncf_is_enabled():
    if not is_nncf_enabled():
        raise RuntimeError('Tried to use NNCF, but NNCF is not installed')


def get_nncf_version():
    if not is_nncf_enabled():
        return None
    return nncf.__version__


if is_nncf_enabled():
    try:
        from nncf.torch.checkpoint_loading import load_state
        from nncf.torch.dynamic_graph.context import get_current_context
        from nncf.torch.dynamic_graph.context import \
            no_nncf_trace as original_no_nncf_trace
    except ImportError:
        raise RuntimeError(
            'Cannot import the standard functions of NNCF library '
            '-- most probably, incompatible version of NNCF. '
            'Please, use NNCF version pointed in the documentation.')


def load_checkpoint(model, filename, map_location=None, strict=False):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Either a filepath or URL or modelzoo://xxxxxxx.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = torch.load(filename, map_location=map_location)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
    _ = load_state(model, state_dict, strict)
    return checkpoint


@contextmanager
def nullcontext():
    """
    Context which does nothing
    """
    yield


def no_nncf_trace():
    """
    Wrapper for original NNCF no_nncf_trace() context
    """

    if is_nncf_enabled():
        return original_no_nncf_trace()
    return nullcontext()


def is_in_nncf_tracing():
    if not is_nncf_enabled():
        return False

    ctx = get_current_context()

    if ctx is None:
        return False
    return ctx.is_tracing
