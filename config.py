import argparse


def get_params():
    parser = argparse.ArgumentParser(description="Parameters based on the Rainbow paper")

    parser.add_argument("--mem_size", default=55000, type=int, help="The memory size")
    parser.add_argument("--env_name", default="MiniGrid-Empty-6x6-v0", type=str, help="Name of the environment")
    parser.add_argument("--interval", default=150, type=int,
                        help="The interval specifies how often different parameters should be saved and printed,"
                             " counted by episodes")
    parser.add_argument("--do_train", action="store_false",
                        help="The flag determines whether to train the agent or play with it")
    parser.add_argument("--train_from_scratch", action="store_false",
                        help="The flag determines whether to train from scratch or continue previous tries")
    parser.add_argument("--weights_path", default="models/params.pth", type=str,
                        help="Path of weights either for train or play ")
    parser.add_argument("--do_intro_env", action="store_true",
                        help="Only introduce the environment then close the program")
    parser_params = parser.parse_args()

    default_params = {"lr": 2.5e-4,
                      "n_step": 3,
                      "batch_size": 32,
                      "gamma": 0.99,
                      "tau": 0.001,
                      "train_period": 4,
                      "v_min": -10,
                      "v_max": 10,
                      "n_atoms": 51,
                      "adam_eps": 0.01 / 32,
                      "alpha": 0.5,
                      "beta": 0.4,
                      "epsilon": 1.0,
                      "decay_rate": 4e-3,
                      "min_epsilon": 0.01
                      }
    total_params = {**vars(parser_params), **default_params}
    print("params:", total_params)
    return total_params
