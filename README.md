# Rainbow DQN 🌈

*Rainbow DQN implementation that outperforms the paper's (Hessel, et al. 2017) results on 40% of games while using 20x less data.* This was developed as part of an undergraduate university course on scientific research and writing. The accompanying paper including results on the ALE can be found [here](). The results are also available as a spreadsheet [here](https://docs.google.com/spreadsheets/d/1ncCFIno4o83JmosAwj30XvIfWSIbO5btomfTrzEr4xE).

### Key Changes and Results
- We implemented all components apart from distributional RL (we saw mixed results with C51 and QR-DQN).
- Integrations and recommended preprocessing for >1000 environments from [gym](https://github.com/openai/gym), [gym-retro](https://github.com/openai/retro) and [procgen](https://github.com/openai/procgen) are provided.
- We implemented the large IMPALA CNN with 2x channels from Espeholt et al. (2018)
- To reduce training time, the implementation uses large, vectorized environments, asynchronous environment interaction, mixed precision training and larger batch sizes.
- Due to compute and time constraints we only trained for 10M frames (compared to 200M in the paper)

When trained for only 10M frames, this implementation outperforms:

| | | |
|-----------------------|:-------------------------|:---------------|
| google/dopamine       | trained for 10M frames  | on 96% of games |
| google/dopamine       | trained for 200M frames | on 64% of games |
| Hessel, et al. (2017) | trained for 200M frames | on 40% of games |
| Human results         |                           | on 72% of games |

Most of the observed performance improvements compared to the paper come from switching to the IMPALA CNN as well as some hyperparameter changes (e.g. the 4x larger learning rate).

### Setup

Install necessary prerequisites with

```
pip install wandb gym[atari] imageio moviepy torchsummary tqdm rich procgen gym-retro torch stable_baselines3
```

If you intend to use `gym` Atari games you will need to install these separately, e.g. by running:

```
wget http://www.atarimania.com/roms/Roms.rar 
unrar x Roms.rar
python -m atari_py.import_roms .
```

For `gym-retro` you will need to follow the instructions [here](https://retro.readthedocs.io/en/latest/getting_started.html#importing-roms).

### How to use

To get started right away, simply run

```
python train_rainbow.py --env_name gym:Qbert
```

This will train Rainbow on Atari Qbert and log all results to "Weights and Biases" as well as the checkpoints directory.

For more configuration options please take a look at `common/argp.py` or run `python train_rainbow.py --help`.

### Some Notes
- With a single RTX 2080 and 12 CPU cores, training for 10M frames takes around 8-12 hours, depending on the used settings
- About 15GB RAM are required. When using a larger replay buffer or subprocess envs, memory use may be *much* higher
- Hyperparameters can be configured through command line arguments, defaults can be found in `common/argp.py`
- For fastest training throughput use `batch_size=512, parallel_envs=64, train_count=1, subproc_vecenv=True`
