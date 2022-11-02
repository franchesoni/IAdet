from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS

@HOOKS.register_module()
class ResetTrainDataloaderHook(Hook):
    def before_train_epoch(self,
                         runner: Runner,
                         ):
        """
        Loads data again from current annotation file, useful when annotation file is updated during training

        Args:
            runner (:obj:`Runner`): The runner of the training process.
        """
        diff_rank_seed = runner._randomness_cfg.get(
            'diff_rank_seed', False)
        runner.train_loop.dataloader = runner.build_dataloader(
            runner._train_dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
        breakpoint()
