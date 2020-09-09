# wds_example

This project hosts a sample code that demonstrates how to use [WebDataset](https://github.com/tmbdev/webdataset) on video data.
The code includes the implementation of the Facebook [SlowFast](https://github.com/facebookresearch/SlowFast) network and the dataloader of [Kinetics-Sounds](https://arxiv.org/abs/1705.08168).

### Environment setup
You can create a conda environment by
```
conda create -y -n wds python=3.7.6
```
and activate it with
```
conda activate wds
```
Install the required packages as follows:
```
# Install via conda-forge for ffmpeg
conda install -y -c conda-forge av
bash install.sh
```

### Kinetics-Sounds
* Create a directory named `datasets`.
* Download and decompress Kinetics-Sounds([download link](https://drive.google.com/file/d/1sqSyNCGLLisl4vnlBiWRGtoFmE1mE2Rq/view?usp=sharing)) into `datasets/KineticsSounds`

## Runing Experiments
Details of experiment configuration are in `config.py`.
To run an experiment
```
python run.py --cfg_file SLOWFAST_8x8_R50.yaml
```

You can manually evaluate a checkpoint
```
python run.py --cfg_file SLOWFAST_8x8_R50.yaml --test TRAIN.ENABLE False TEST.ENABLE True TEST.CHECKPOINT_FILE_PATH [path-to-checkpoint]
```


### Note
* As of September 4, 2020, using `MultiDataset` and `ResizedDataset` simultaneously gives rise to an AttributeError ([issue](https://github.com/tmbdev/webdataset/issues/9)). Thus, please DO NOT USE multiprocessing for data loading (that is, `DATA_LOADER.NUM_WORKERS = 0`).
* If the above issue is solved: when using multiprocessing for data loading, the number of workers cannot exceed the number of shard `.tar` files.
