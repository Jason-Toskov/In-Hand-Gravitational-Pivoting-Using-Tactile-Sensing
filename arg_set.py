import argparse
from pathlib import Path

# simple function get all args from input
# Can be used to do wandb sweeps as well as getting args from a config file
def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c', '--config', type=Path, default=Path('./param_config.yml'))
    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--test_batch_size', type=int)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--label_scale_position', type=int)
    parser.add_argument('--label_scale_velocity', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--plot_path', type=str)
    parser.add_argument('--resume_from_checkpoint', type=bool)
    parser.add_argument('--test_only', type=bool)
    parser.add_argument('--train_frac', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--sample', type=str)
    parser.add_argument('--data_mode', type=str)
    parser.add_argument('--use_wandb', type=bool)
    parser.add_argument('--wandb_run_notes', type=str)
    parser.add_argument('--MLP_window', type=int)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--wandb_project', type=str)
    parser.add_argument('--wandb_name', type=str, default='None')

    
    args = parser.parse_args()
    
    return args