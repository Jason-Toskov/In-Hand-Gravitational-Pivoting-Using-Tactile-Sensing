# In-Hand Gravitational Pivoting Using Tactile Sensing

Code for training models from the paper In-Hand Gravitational Pivoting Using Tactile Sensing.

OpenReview: [LINK](https://openreview.net/forum?id=NEGjAH7p0fm) 

ArXiv: [LINK](https://arxiv.org/abs/2210.05068)

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

The full dataset can be downloaded from Monash Bridges, [linked here](https://bridges.monash.edu/articles/dataset/Dataset_for_Gravitational_Pivoting/21482841).

After the data has been loaded, ensure it is moved to the same folder as this repo and unzip it using the following command:

`unzip dataset.zip -d training_data/`

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

## Citation

If you find our work or dataset useful, please cite us:

```
@inproceedings{toskovGravitationalPivoting,
  title={In-Hand Gravitational Pivoting Using Tactile Sensing},
  author={Toskov, Jason and Newbury, Rhys and Mukadam, Mustafa and Kulić, Dana and Cosgun, Akansel},
  year={2022},
}
```
