import argparse


def get_params():
    parser = argparse.ArgumentParser(description="Parameters based on the Rainbow paper")
    parser.add_argument("--lr", default=6.25e-5, type=float, help="The learning rate")
    parser.add_argument("--multi_step_n", default=3, type=int, help="The number of step to take account for"
                                                                    "multi step learning")
    parser.add_argument("--batch_size", default=32, type=int, help="The batch size")
    parser.add_argument("--mem_size", default=12000, type=int, help="The memory size")
    parser.add_argument("--gamma", default=0.99, type=float, help="The discount factor")
    parser.add_argument("--tau", default=0.001, type=float, help="Soft update exponential rate")
    parser.add_argument("--max_episodes", default=10000, type=int, help="Maximum number of episodes to train the agent")
    parser.add_argument("--env_name", default="Breakout-v0", type=str, help="Name of the environment")
    parser.add_argument("--log_interval", default=1, type=int, help="The interval specifies how often different metrics"
                                                                    "should be logged, counted by episodes")
    parser.add_argument("--save_interval", default=200, type=int, help="The interval specifies how often different"
                                                                       "parameters should be saved, counted by episodes")
    parser.add_argument("--print_interval", default=200, type=int, help="The interval specifies how often different"
                                                                        "parameters should be printed, counted by episodes")
    parser.add_argument("--train_period", default=4, type=int,
                        help="The period that specifies the number of steps which"
                             " the networks are not updated")
    parser.add_argument("--do_train", action="store_false", help="The flag determines whether to train"
                                                                 "the agent or play with it")
    params = parser.parse_args()
    return vars(params)
