DATA_DIR="$1"
WIDTH=500
python IAdet.py --data_dir $DATA_DIR --width $WIDTH -v & gui_pid=$!;
echo "Started GUI with PID ${gui_pid}"
echo "Waiting for annotation file..."
while [ ! -f annotated_iadet.json ]; do sleep 1; done  # wait for an annotation file to be generated
python mmdetection/tools/train.py model/faster_iadet.py & mmdet_pid=$!;
echo "Started background model training with PID ${mmdet_pid}"
