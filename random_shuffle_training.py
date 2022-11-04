# References:
# https://stackoverflow.com/a/12471272

import yaml
import os
import shutil
import trainer
import random

data_path = './train_test_split_storage/'
input_dir = './training_data/'

model_list = ['LSTM', 'MLP']
epoch_list = [100, 100]
pred_type = ['both']

train_frac = 0.8
num_trials = 5

for model, epochs in zip(model_list, epoch_list):
    for pred in pred_type:
        for i in range(num_trials): # number of different trials
            with open('param_config.yml', 'r') as f:
                config_file = yaml.safe_load(f)
            
            config_file['wandb_name'] = 'RandomShuffle-'+pred+'-'+model+'-'+str(i)
            config_file['model_type'] = model
            config_file['num_epochs'] = epochs
            config_file['model_prediction'] = pred

            # https://stackoverflow.com/a/12471272
            with open('param_config.yml', 'w') as f:
                yaml.dump(config_file, f, default_flow_style=False)

            if not os.path.exists(data_path):
                os.makedirs(data_path)
            else:
                shutil.rmtree(data_path)
                os.makedirs(data_path)
            
            os.makedirs(data_path+'train/')
            os.makedirs(data_path+'test/')

            directories = os.listdir(input_dir)
            random.Random(i).shuffle(directories)

            max_train_idx = int(round(len(directories)) * train_frac)
            train_files = directories[:max_train_idx]
            test_files = directories[max_train_idx:]

            for f_train in train_files:
                shutil.copyfile(input_dir+f_train, data_path+'train/'+f_train)

            for f_test in test_files:
                shutil.copyfile(input_dir+f_test, data_path+'test/'+f_test)

            print("Num train samples:", len(os.listdir(data_path+'train/')))
            print("Num test samples:", len(os.listdir(data_path+'test/')))

            trainer.main()

