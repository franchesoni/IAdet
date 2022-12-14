
_base_ = [
  "../mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py", 
  "../mmdetection/configs/_base_/schedules/schedule_1x.py", 
  "../mmdetection/configs/_base_/default_runtime.py", 
  "iadet_data.py", 
]

custom_hooks=[
  dict(
  type="ConvertToBaseDetHook", priority="ABOVE_NORMAL"
),
  dict(
  type="ResetTrainDataloaderHook", priority="BELOW_NORMAL"
)]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10000, val_interval=20000)
checkpoint_config = dict(max_keep_ckpts = 1)