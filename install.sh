pip install --upgrade pip
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc1"
git clone https://github.com/open-mmlab/mmdetection.git -b 3.x
cd mmdetection
cp ../modifications_to_mmdet/* mmdet/engine/hooks/
pip install -v -e .