# References:
# https://stackoverflow.com/a/12471272

import yaml
import os
import shutil
import trainer
import random

data_path = './train_test_split_storage/'
input_dir = './training_data/'

objects_in_classes = {
    'Box': ['breadboard','earbud','magnet','toothbrush','toothpaste'],
    'Cylinder': ['deodorant','pill','shampoo','spray','swiss'],
}
model_list = ['LSTM', 'MLP']
epoch_list = [100, 100]

train_frac = 0.8

for model, epochs in zip(model_list, epoch_list):
    for cls in objects_in_classes:
        with open('param_config.yml', 'r') as f:
            config_file = yaml.safe_load(f)
        
        config_file['wandb_name'] = 'TestOnOtherClass-'+model+'-TrainOn'+cls
        config_file['model_type'] = model
        config_file['num_epochs'] = epochs

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

        train_files = [x for x in os.listdir(input_dir) if any(obj in x for obj in objects_in_classes[cls])]
        test_files = [x for x in os.listdir(input_dir) if not any(obj in x for obj in objects_in_classes[cls])]

        for f_train in train_files:
            shutil.copyfile(input_dir+f_train, data_path+'train/'+f_train)

        for f_test in test_files:
            shutil.copyfile(input_dir+f_test, data_path+'test/'+f_test)

        print("Num train samples:", len(os.listdir(data_path+'train/')))
        print("Num test samples:", len(os.listdir(data_path+'test/')))

        trainer.main()

        with open('new_config.yml', 'r') as f:
            config_file = yaml.safe_load(f)
        
        config_file['wandb_name'] = 'TestOnSameClass-'+model+'-'+cls
        config_file['model_type'] = model
        config_file['num_epochs'] = epochs

        # https://stackoverflow.com/a/12471272
        with open('new_config.yml', 'w') as f:
            yaml.dump(config_file, f, default_flow_style=False)

        if not os.path.exists(data_path):
            os.makedirs(data_path)
        else:
            shutil.rmtree(data_path)
            os.makedirs(data_path)
        
        os.makedirs(data_path+'train/')
        os.makedirs(data_path+'test/')

        directories = [x for x in os.listdir(input_dir) if any(obj in x for obj in objects_in_classes[cls])]
        random.Random(0).shuffle(directories)

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

