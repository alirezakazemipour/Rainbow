[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)  
# Rainbow
Implemetation of the rainbow paper.  
![Pong](https://github.com/alirezakazemipour/DQN_Based_Agents/blob/master/Results/rainbow.gif)    
#
![running_reward](https://github.com/alirezakazemipour/DQN_Based_Agents/blob/master/Results/running_reward.png)
![last_10_mean_reward](https://github.com/alirezakazemipour/DQN_Based_Agents/blob/master/Results/10_last_mean_reward.png)  
## Demo
<p align="center">
  <img src="Results/rainbow.gif" height=250>
</p>
## Dependencies
- gym == 0.17.2
- numpy == 1.19.1
- opencv_contrib_python == 3.4.0.12
- psutil == 5.4.2
- torch == 1.4.0

## Installation
```shell
pip3 install -r requirements.txt
```
## Usage
```bash
main.py [-h] [--algo ALGO] [--mem_size MEM_SIZE] [--env_name ENV_NAME]
               [--interval INTERVAL] [--do_train] [--train_from_scratch]
               [--do_intro_env]

Variable parameters based on the configuration of the machine or user's choice

optional arguments:
  -h, --help            show this help message and exit
  --algo ALGO           The algorithm which is used to train the agent.
  --mem_size MEM_SIZE   The memory size.
  --env_name ENV_NAME   Name of the environment.
  --interval INTERVAL   The interval specifies how often different parameters
                        should be saved and printed, counted by episodes.
  --do_train            The flag determines whether to train the agent or play
                        with it.
  --train_from_scratch  The flag determines whether to train from scratch or[default=True]
                        continue previous tries.
  --do_intro_env        Only introduce the environment then close the program.
```