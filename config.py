import argparse


def get_params():
    parser = argparse.ArgumentParser(description="Parameters based on the Rainbow paper")
    parser.add_argument("--lr", default=2.5e-4, type=float, help="The learning rate")
    parser.add_argument("--multi_step_n", default=3, type=int,
                        help="The number of step to take account for multi step learning")
    parser.add_argument("--batch_size", default=32, type=int, help="The batch size")
    parser.add_argument("--mem_size", default=25000, type=int, help="The memory size")
    parser.add_argument("--gamma", default=0.99, type=float, help="The discount factor")
    parser.add_argument("--tau", default=0.001, type=float, help="Soft update exponential rate")
    parser.add_argument("--max_episodes", default=100000, type=int, help="Maximum number of episodes to train the agent")
    parser.add_argument("--env_name", default="MsPacmanNoFrameskip-v4", type=str, help="Name of the environment")
    parser.add_argument("--interval", default=50, type=int,
                        help="The interval specifies how often different parameters should be saved and printed,"
                             " counted by episodes")
    parser.add_argument("--train_period", default=4, type=int,
                        help="The period that specifies the number of steps which the networks are not updated")
    parser.add_argument("--do_train", action="store_false", help="The flag determines whether to train"
                                                                 "the agent or play with it")
    parser.add_argument("--train_from_scratch", action="store_false", help="The flag determines whether to train"
                                                                        "from scratch or continue previous tries")
    parser.add_argument("--V_min", default=-10, type=int, help="Lower bound of the value estimation of"
                                                               "the distributional algorithm")
    parser.add_argument("--V_max", default=10, type=int, help="Upper bound of the value estimation of"
                                                              "the distributional algorithm")
    parser.add_argument("--N_atoms", default=51, type=int, help="Number of atoms to predict the value distribution in"
                                                                "the distributional algorithm")
    parser.add_argument("--adam_eps", default=1.5e-4, type=float, help="The Adam epsilon")
    parser.add_argument("--alpha", default=0.5, type=float, help="The aggressiveness parameter of PER")
    parser.add_argument("--beta", default=0.4, type=float, help="The robustness parameter of PER")
    parser.add_argument("--epsilon", default=1.0, type=float, help="Exploration probability")
    parser.add_argument("--decay_rate", default=4e-3, type=float, help="Decay rate of the exploration probability")
    parser.add_argument("--min_epsilon", default=0.01, type=float, help="Minimum probability of exploration")
    parser.add_argument("--weights_path", default="models/params.pth", type=str, help="Path of weights either for"
                                                                                      "train or play ")
    parser.add_argument("--do_intro_env", action="store_true", help="Only introduce the environment then close the "
                                                                    "program")
    params = parser.parse_args()
    print("params:", vars(params))
    return vars(params)
