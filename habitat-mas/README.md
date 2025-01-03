# Habitat-MAS Package

Habitat-MAS is a Python package for Multi-Agent Systems in Habitat virtual environments.

## Table of Contents
- [Habitat-MAS Package](#habitat-mas-package)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Install habitat-mas](#install-habitat-mas)
  - [Usage](#usage)
    - [Download Data](#download-data)
      - [Download Base Dataset](#download-base-dataset)
      - [Download MP3D Dataset](#download-mp3d-dataset)
      - [Download Habitat-MAS Dataset](#download-habitat-mas-dataset)
    - [Run the demo](#run-the-demo)
    - [Run Habitat-MAS](#run-habitat-mas)
      - [Perception](#perception)
      - [Mobility](#mobility)
      - [Manipulation](#manipulation)
      - [Rearrange](#rearrange)
    - [Dataset Structure](#dataset-structure)
    - [Dataset Generation](#dataset-generation)

## Installation

### Prerequisites

Please make sure you have installed the [habitat-sim](https://github.com/facebookresearch/habitat-sim/tree/v0.3.1), [habitat-lab](../README.md) and [habitat-baselines](../habitat-baselines/) following the installation guide in the previous step. 

**Note**: If you try to install the habitat suites on a Linux server without GUI, you may need to install headless version of the habitat-sim:

```sh
# Make sure nvidia-smi is available on your linux server
# $ sudo apt list --installed | grep nvidia-driver
# > nvidia-driver-xxx ...

conda install habitat-sim withbullet headless -c conda-forge -c aihabitat
```

### Install habitat-mas

To install the package, you can use the following command under habitat-lab root directory:

```sh
pip install -e habitat-mas
```

## Usage

### Download Data

#### Download Base Dataset
The dataset used in the demo is the same as [Habitat-3.0 Multi-Agent Training](../habitat-baselines/README.md#habitat-30-multi-agent-training). You can download the dataset by running the following command:

```sh
python -m habitat_sim.utils.datasets_download --uids hssd-hab hab3-episodes habitat_humanoids hab_spot_arm hab3-episodes ycb hssd-hab hab3_bench_assets rearrange_task_assets
```
#### Download MP3D Dataset
Since the mobility and rearrange tasks utilize the scene data of [Matterport3D](https://niessner.github.io/Matterport/), it is necessary to download the scene data of mp3d-habitat first according to the instructions in [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#matterport3d-mp3d-dataset).

In short, you should first fill and sign a form to get `download_mp.py`, then run the following command in Python 2.7:
```sh
python download_mp.py --task habitat
```
Remeber to rename the folder name from 'mp3d-habitat' to 'mp3d', the path of MP3D dataset will be `EMOS/data/scene_datasets/mp3d/...`.

#### Download Habitat-MAS Dataset
Besides, you should:
Download the robot configuration and episodes data from [Here](https://drive.google.com/drive/folders/1YVoCg2-tGkKWrdej4km6Abxsop0wS9XJ?usp=drive_link), extract and merge it into EMOS like `EMOS/data/...`.

The folder should look like this:
```
EMOS
├── data
│   ├── robots
│       ├── dji_drone
│           ├── meshes
│           ├── urdf
│       ├── hab_fetch
│       ├── hab_spot_arm
|           ├── meshesDae
│           ├── urdf
|               ├── hab_spot_onlyarm_dae.urdf
│               ...
│       ├── hab_stretch
│       ├── spot_data
│           ├── spot_walking_trajectory.csv
│       ├── robot_configs
│           ├── hssd
│             ├── hssd_eval.json
│           ├── mp3d
│             ├── mobility_eval.json
│           ├── replica_cad
│             ├── two_agent_perception_eval.json
│             ...
│       ...
│   ├── datasets
│       ├── replica_cad
│           ├── single_agent_eval.json.gz
│           ...
│       ├── hssd
│           ├── hssd_eval.json.gz
│           ...
│       ├── mp3d
│           ├── mobility_eval.json.gz
│           ...
...
```

### Run the demo

The demo is adapted from [Habitat-3.0 Social Rearrangement](../habitat-baselines/README.md#social-rearrangement). You can run the demo by running the following command:

```sh
# Under the habitat-lab root directory
python -u -m habitat_baselines.run \
  --config-name=social_rearrange/llm_spot_drone.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1
```

### Run Habitat-MAS

To run our episodes in Habitat-MAS, you should first set your API key in [`habitat-mas/habitat_mas/utils/models.py`](https://github.com/SgtVincent/EMOS/blob/8f4348d73fcf605ebfbeee13ff897359723b5f1c/habitat-mas/habitat_mas/utils/models.py) to run EMOS.

For each task, you could run the following command:

#### Perception
```sh
python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  # --config-name=multi_rearrange/llm_height_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1
```

#### Mobility
```sh
python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_fetch_mobility.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1
```

#### Manipulation
```sh
python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_fetch_stretch_man.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1
```

#### Rearrange
```sh
python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_rearrange.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1
```

### Dataset Structure

**Multi robot perception**
`data/datasets/hssd/0/hssd_height_per.json.gz`
- Mainly deploy robots spot and drone for object perception.
- Objects are easy for drone to find, but hard for spot robot.

**Multi robot manipulation**
`data/datasets/hssd/0/hssd_height_man.json.gz` or `data/datasets/hssd/0/hssd_dist_man.json.gz`

- Mainly deploy robots fetch and stretch for object rearrangement.
- Objects are easy for stretch robot to get, but hard for fetch robot.

**Multi robot mobility**
`data/datasets/mp3d/mobility_episodes_1.json.gz`
- Mainly deploy robots spot and fetch for cross-floor navigation.
- Cross-floor tasks are easy for spot to do, but impossible for fetch robot.

**Multi robot rearrange**
`data/datasets/mp3d/mp3d_episodes_1.json.gz`
- Mainly deploy spot, fetch and drone for cross-floor rearrangement.
- Cross-floor rearrangement is suitable for spot, same-floor rearrangement is suitable for fetch, cross-floor perception is suitable for drone.

### Dataset Generation

Besides the dataset we have provided, custom data can be also generated through the scripts we provided.

Here is a demo data generation command for dataset in `hssd` scene.

```sh
python habitat-lab/habitat/datasets/rearrange/run_hssd_episode_generator.py --run --config data/config/hssd/hssd_dataset.yaml --num-episodes 340 --out data/datasets/hssd_height.json.gz --type height
```

`--config`: path of your dataset generation configuration data.

`--num-episodes`: episodes number you want to generate. (Note: in `scene_balanced` type `scene_sampler`, the number should be Integer multiple of `34`, you can customize your dataset generation configuration, which should be the file in the path of `--config`)

`--out`: desired path of your newly generated dataset.

`--type`: the purpose of your dataset, currently there are three types: `height`, `distance`, `normal`.
