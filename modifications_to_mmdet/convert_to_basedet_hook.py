import os

from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS
from mmengine import load, dump

def iadet2mmdet(annotations_iadet: dict) -> list[dict]:
    """Converts from iadet format (dict of dicts) to iadet format (list of dicts).
    Allows for more operation with mmdetection."""
    return {"metainfo":{}, "data_list":list(annotations_iadet.values())}



@HOOKS.register_module()
class ConvertToBaseDetHook(Hook):
    def before_train_epoch(self,
                         runner: Runner,
                         ):
        """
        Converts annotation in IAdet format to mmdetection's BaseDet format

        Args:
            runner (:obj:`Runner`): The runner of the training process.
        """
        in_path = os.path.join(runner._train_dataloader['dataset']['dataset']['data_root'],
        runner._train_dataloader['dataset']['dataset']['ann_file'].replace('mmdet', 'iadet'))
        out_path = os.path.join(runner._train_dataloader['dataset']['dataset']['data_root'],
        runner._train_dataloader['dataset']['dataset']['ann_file'])
        anniadet = load(in_path)
        annmmdet = iadet2mmdet(anniadet)
        dump(annmmdet, out_path)

