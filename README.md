# Rainbow 🌈

*An variant of Rainbow DQN which reaches a median HNS of 205.7 after only 10M frames (the original Rainbow from Hessel et al. 2017 reached 231.0 using 20x more data).* See the [paper](https://arxiv.org/abs/2111.10247) for more details. This was developed as part of an undergraduate university course on scientific research and writing. A selection of videos is available [here](https://drive.google.com/drive/folders/1bNRyHcDYxSbww1aGskhqoMA2OurJXtOU).

### Key Changes and Results
- We used the large IMPALA-CNN with 2x channels from Espeholt et al. (2018), other networks are also implemented.
- We used spectral normalization in the residual blocks which resulted in faster learning (especially at the start of training).
- We removed the distributional RL component since we didn't see any benefit when only training for 10M frames and appreciated the reduced implementation complexity (we tried both C51 and QR-DQN).
- We performed additional hyperparameters tuning (see paper).
- The implementation uses large, vectorized environments, asynchronous environment interaction, mixed-precision training, and larger batch sizes to improve computational efficiency and reduce training time.
- Integrations and recommended preprocessing for >1000 environments from [gym](https://github.com/openai/gym), [gym-retro](https://github.com/openai/retro) and [procgen](https://github.com/openai/procgen) are provided.

Please cite the [paper](https://arxiv.org/abs/2111.10247) if you use this implementation in your publication.

### Setup

Install necessary prerequisites with

```
sudo apt install zlib1g-dev cmake unrar
pip install wandb gym[atari]==0.18.0 imageio moviepy torchsummary tqdm rich procgen gym-retro torch stable_baselines3 atari_py==0.2.9
```

If you intend to use `gym` Atari games, you will need to install these separately, e.g., by running:

```
wget http://www.atarimania.com/roms/Roms.rar 
unrar x Roms.rar
python -m atari_py.import_roms .
```

To set up `gym-retro` games you should follow the instructions [here](https://retro.readthedocs.io/en/latest/getting_started.html#importing-roms).

### How to use

To get started right away, run

```
python train_rainbow.py --env_name gym:Qbert
```

This will train Rainbow on Atari Qbert and log all results to "Weights and Biases" and the checkpoints directory.

Please take a look at `common/argp.py` or run `python train_rainbow.py --help` for more configuration options.

### Some Notes
- With a single RTX 3090 and 12 CPU cores, training for 10M frames takes around 7.5 hours.
- About 15GB of RAM are required. When using a larger replay buffer or subprocess envs, memory use may be *much* higher.
- Hyperparameters can be configured through command line arguments; defaults can be found in `common/argp.py`
- For highest training throughput use `batch_size=512, parallel_envs=64, train_count=1, subproc_vecenv=True`

### Acknowledgements

We are very grateful to the TU Wien DataLab for providing the majority of the compute resources that were necessary to perform the experiments.

Here are some other implementations and resources that were helpful in the completion of this project:
- OpenAI Baselines (especially for preprocessing and Atari wrappers)
- https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py
- https://github.com/Kaixhin/Rainbow/
- https://github.com/Kaixhin/Rainbow/wiki/Matteo's-Notes
