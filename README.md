# Architect: Generating Vivid and Interactive 3D Scenes with Hierarchical 2D Inpainting

#### This repo is still under development.

## Installation
Our environment is running under CUDA 12.1.

You can create a Conda environment for this simulator first:
```bash
conda create -n architect python=3.9.16
conda activate arthitect

git clone https://github.com/wangyian-me/architect_official_code.git
cd architect_official_code
```

Our work have dependency on the following amazing projects:

[Genesis](https://github.com/Genesis-Embodied-AI/Genesis)

[Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main)

[DINO-X](https://github.com/IDEA-Research/DINO-X-API/tree/main)

[Marigold](https://github.com/prs-eth/Marigold.git)

[Stable-diffusion-xl-inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)

And optionally,
[t2v_metrics](https://github.com/linzhiqiu/t2v_metrics).

Set up these environments accordingly and add Grounded-SAM (mainly SAM since we can replace the Grounded-DINO with DINO-X, we will replace the SAM with DINO-X api soon), Marigold and t2v_metrics to `.../architect_official_code/` using soft link. It is tested that using the latest torch, huggingface, diffusers, transformers version will work.

## Dataset

Meanwhile, we also need to build the dataset to retrieve objects. Specifically, we build a pipeline to automatically download and label assets from [Blenderkit](https://www.blenderkit.com/), where you can find your API Key in this [page](https://www.blenderkit.com/profile/addon/) after login.
To download from Blenderkit, we need to create another conda environment with `bpy`.

```bash
conda create -n blenderkit python=3.11
conda activate blenderkit
pip install numpy requests OpenEXR trimesh Imath tqdm opencv-python
pip install bpy==4.2.0
```

We provide a set of annotated objects together with the encoded features for retrieval usage.
To download an asset with a `uid` you can either run:
```bash
conda activate blenderkit
cd build_dataset_architect
python download_blenderkit.py --uid {uid} --save_dir {save_dir} --api_key {api_key}
```

or download in python code,
```python
target_env = os.environ.get("BLENDER_ENV")
target_code = 'download_blenderkit.py'
subprocess.run([
          'conda', 'run', 
          '--prefix', target_env, 
          'python', target_code,
          '--uid', uid,
          '--save_dir', save_dir
      ])
```


After setting up the conda environment of blenderkit, assume it's at `{your_conda_path}/envs/blenderkit`. And assume you want to download your data in `architect_official_code/data/blenderkit_data_annotated`. Run the following code to start to build dataset.

```bash
cd architect_official_code
mkdir data
cd data
mkdir blenderkit_data_annotated
cd ../../build_dataset_architect

conda activate arthitect

export BLENDERKIT_API_KEY=YOUR_API_KEY
export BLENDER_ENV={your_conda_path}/envs/blenderkit
export OPENAI_API_KEY=Your OPENAI_API_KEY
export DATASET_PATH=......./architect_official_code/data/blenderkit_data_annotated

python main.py
```


## Usage example

before running architect code, run these scripts to set up paths:
```bash
cd architect_official_code
cd data
mkdir cache
cd ..

conda activate arthitect

export OPENAI_API_KEY=Your OPENAI_API_KEY
export DATASET_PATH=......./architect_official_code/data/blenderkit_data_annotated
export CACHE_PATH=......./architect_official_code/data/cache
export DINO_API_KEY=Your DINO_API_KEY
```


To generate large furniture layout you can run:
```bash
python pipeline_room.py --work_dir ./data/architect_results --room_x 6 --room_y 6 --wall_texture_dir 1.jpg --floor_texture_dir 2.jpg --prompt "living room" --scene_name "living_room_0"
```
With large furniture layout, you can run the following scripts to generate small object placements:
```bash
python pipeline_room_small.py --work_dir ./data/architect_results --room_x 6 --room_y 6 --wall_texture_dir 1.jpg --floor_texture_dir 2.jpg --prompt "living room" --scene_name "living_room_0"
```

## Citation
If you find this codebase/paper useful for your research, please consider citing:
```
@inproceedings{
  wang2024architect,
  title={Architect: Generating Vivid and Interactive 3D Scenes with Hierarchical 2D Inpainting},
  author={Yian Wang and Xiaowen Qiu and Jiageng Liu and Zhehuan Chen and Jiting Cai and Yufei Wang and Tsun-Hsuan Wang and Zhou Xian and Chuang Gan},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=JHg9eNuw6p}
}
```

