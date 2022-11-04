# In-Hand Gravitational Pivoting Using Tactile Sensing

Code for training models from the paper In-Hand Gravitational Pivoting Using Tactile Sensing.

## Requirements

- Python 3
- PyTorch
- NumPy
- Pandas
- Matplotlib
- PyYAML
- WandB
- Plotly

All requirements can be installed using `pip install -r requirements.txt`.

## Dataset

For anonymous review a random sample of 400 datapoints from the full dataset is included due to data storage limitations. Links to the full dataset will be provided after anonymous review.

## Training

WandB should first be activated by running `wandb login` and following the prompts. WandB is required for plots of sample sequence predictions and will also store models in an easily recoverable manner. This code can be ran without WandB by setting `use_wandb: False` in `param_config.yml`.

Ensure data is then located in a folder named `./training_data/`. The 3 experiments reported in the paper can then be ran using:

- `python random_shuffle_training.py`: Rotation estimation random split
- `python held_out_object_training.py`: Unseen objects
- `python held_out_class_training.py`: Unseen classes

Models can also be trained by ensuring the data is located in a folder named `./train_test_split_storage/` and running `python trainer.py`, with the desired parameters for the experiment being set in `param_config.yml`.

## Testing 

Testing can be performed on trained models by setting various parameters in `param_config.yml`. The required params are as follows:

- `test_only: True`
- `model_path: <Saved model path>`
- `resume_from_checkpoint: True`

## References

- https://stackoverflow.com/a/3114573
- https://www.geeksforgeeks.org/python-k-middle-elements/
- https://stackoverflow.com/a/12471272
