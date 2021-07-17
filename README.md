# Rainbow DQN 🌈

- Implementation of the "Rainbow DQN" algorithm presented in Hessel, et al. (2017).
- Includes all components apart from distributional RL (we saw mixed results with C51 and QR-DQN).
- The training framework supports >1000 environments from gym, gym-retro and procgen.
- By default the large IMPALA CNN with 2x channels from Espeholt et al. (2018) is used.
- To reduce training time, the implementation uses large, vectorized environments and larger batch sizes.

When trained for 10M frames, this implementation outperforms
  * google/dopamine           (trained for  10M frames)       on 96% of games
  * google/dopamine           (trained for 200M frames)       on 64% of games
  * Hessel, et al. (2017)     (trained for 200M frames)       on 40% of games
  * Human results                                             on 72% of games

Most of the performance improvements compared to the paper come from the IMPALA CNN as well as some 
hyperparameter changes (e.g. 4x larger learning rate).
Full results available here: https://docs.google.com/spreadsheets/d/1ncCFIno4o83JmosAwj30XvIfWSIbO5btomfTrzEr4xE

Some notes & tips
- With a single RTX 3090 and 12 CPU cores, training for 10M frames takes around 8-10 hours, depending on the used settings
- About 15GB RAM are required. When using a larger replay buffer, subprocess envs or larger resolutions, memory use may be *much* higher
- hyperparameters can be configured through command line arguments, defaults are saved in `common/argp.py`
- for fastest training throughput use batch_size=512, parallel_envs=64, train_count=1, subproc_vecenv=True
