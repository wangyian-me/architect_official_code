# Architect: Generating Vivid and Interactive 3D Scenes with Hierarchical 2D Inpainting

#### This repo is still under development.

## Installation
You can create a Conda environment for this simulator first:
```bash
conda create -n architect python=3.9.16
conda activate arthitect
```

And install the package with its dependencies using
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

