try:
    from mmengine import Registry
except ImportError:
    from mmcv import Registry


MODELS = Registry('models')
