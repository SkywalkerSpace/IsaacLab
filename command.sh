export UV_DEFAULT_INDEX=https://mirrors.cloud.tencent.com/pypi/simple/
export UV_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple/

uv run --extra teleop,isaacsim isaaclab teleop run \
--task IsaacContrib-PickPlace-GR1T2-GR00T-Abs-v0 --visualizer kit --xr \
--enable_debug_visualization --disable_external_cameras


uv run --extra teleop,isaacsim isaaclab teleop record \
--task IsaacContrib-PickPlace-GR1T2-GR00T-Abs-v0 --visualizer kit --xr \
--device cpu --num_demos 5 --dataset_file ./datasets/dataset_gr1_vr.hdf5 \
--enable_debug_visualization --disable_external_cameras


uv run --extra teleop,isaacsim isaaclab teleop replay \
--task IsaacContrib-PickPlace-GR1T2-GR00T-Abs-v0 --visualizer kit \
--dataset_file ./datasets/dataset_gr1_vr.hdf5 


uv run --extra isaacsim,mimic python scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
--task IsaacContrib-PickPlace-GR1T2-GR00T-Abs-Mimic-v0 \
--visualizer kit \
--device cpu \
--input_file ./datasets/dataset_gr1_vr_50.hdf5 \
--output_file ./datasets/dataset_annotated_gr1_50.hdf5


uv run --extra isaacsim,mimic python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
--device cpu \
--num_envs 40 \
--generation_num_trials 1000 \
--input_file ./datasets/dataset_annotated_gr1_50.hdf5 \
--output_file ./datasets/generated_dataset_gr1i_50.hdf5
