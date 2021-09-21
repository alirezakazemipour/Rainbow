import argparse


def get_params():
    parser = argparse.ArgumentParser(
        description="Variable parameters based on the configuration of the machine or user's choice")
    parser.add_argument("--algo", default="rainbow", type=str,
                        help="The algorithm which is used to train the agent.")
    parser.add_argument("--mem_size", default=55000, type=int, help="The memory size.")
    parser.add_argument("--env_name", default="BreakoutNoFrameskip-v4", type=str, help="Name of the environment.")
    parser.add_argument("--interval", default=10, type=int,
                        help="The interval specifies how often different parameters should be saved and printed,"
                             " counted by episodes.")
    parser.add_argument("--do_train", action="store_true",
                        help="The flag determines whether to train the agent or play with it.")
    parser.add_argument("--train_from_scratch", action="store_true",
                        help="The flag determines whether to train from scratch or continue previous tries.")

    parser.add_argument("--do_intro_env", action="store_true",
                        help="Only introduce the environment then close the program.")
    parser_params = parser.parse_args()
    assert parser_params.algo is not None

    #  Parameters based on the Rainbow paper
    # region default parameters
    default_params = {"lr": 6.25e-5,
                      "n_step": 3,
                      "batch_size": 32,
                      "state_shape": (4, 84, 84),
                      "max_steps": int(1e+8),
                      "gamma": 0.99,
                      "train_period": 4,
                      "v_min": -10,
                      "v_max": 10,
                      "n_atoms": 51,
                      "adam_eps": 1.5e-4,
                      "alpha": 0.5,
                      "beta": 0.4,
                      "clip_grad_norm": 10.0,
                      "final_annealing_beta_steps": int(1e+6),
                      "initial_mem_size_to_train": 1000,
                      "seed": 123,
                      "tau": 1.25e-4
                      }
    # endregion
    total_params = {**vars(parser_params), **default_params}
    print("params:", total_params)
    return total_params
