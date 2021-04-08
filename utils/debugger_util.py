import pdb
import sys
import matplotlib as mpl
import torch

mpl.use('Agg')
import matplotlib.pyplot as plt

def visualize_tensor(x, img_name):
    if len(x.shape) ==3 and x.shape[0] == 3:
        x = x.transpose(0,1).transpose(1, 2)
    x = x.cpu().detach()
    plt.imshow(x)
    plt.savefig(img_name)

def print_gradients(x):
    print('New gradient')
    print('shape', x.shape)
    if (x != x).sum() > 0:
        print('gradient is nan')
        pdb.set_trace()
    print('mean', x.mean())
    print('min', x.min())
    print('max', x.max())
    print('gradient', x)
    pdb.set_trace()

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def is_weight_nan(model):
    all_params = [k for k in model.parameters()]
    # [p.shape for p in all_params]
    has_nan = [torch.any(p != p) for p in all_params]
    norms = [p.norm() for p in all_params]
    print('norms', norms)
    print('total nans', sum(has_nan), 'out of', len(has_nan))
    print('total norm', sum(norms) / len(norms))

