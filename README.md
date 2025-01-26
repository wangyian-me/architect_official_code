# Architect: Generating Vivid and Interactive 3D Scenes with Hierarchical 2D Inpainting

#### This repo is still under development.

## Installation
You can create a Conda environment for this simulator first:
```bash
conda create -n architect python=3.9.16
conda activate arthitect
```

Our work have dependency on the following amazing projects:
[Genesis](https://github.com/Genesis-Embodied-AI/Genesis)

[Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main)

[DINO-X](https://github.com/IDEA-Research/DINO-X-API/tree/main)

[Marigold](https://github.com/prs-eth/Marigold.git)

[Stable-diffusion-xl-inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)

And optionally,
[t2v_metrics](https://github.com/linzhiqiu/t2v_metrics).

Set up these environments accordingly and add Grounded-SAM (mainly SAM since we can replace the Grounded-DINO with DINO-X, we will replace the SAM with DINO-X api soon), Marigold and t2v_metrics to `.../architect/` using soft link. It is tested that using the latest torch, huggingface, diffusers, transformers version will work.

Meanwhile, we also need to build the dataset of object retrievel.
```bash

```

## Usage example

To generate large furniture layout you can run:
```bash
python pipeline_room.py --
```
With large furniture layout, you can run the following scripts to generate small object placements:
```bash
python pipeline_room_small.py --
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

