# References:
# https://stackoverflow.com/a/12471272

import yaml
import os
import shutil
import trainer

data_path = './train_test_split_storage/'
input_dir =  './training_data/'

object_list = ['breadboard','deodorant','earbud','magnet','pill','shampoo','spray','swiss','toothbrush','toothpaste']
model_list = ['LSTM', 'MLP']
epoch_list = [100, 100]

for model, epochs in zip(model_list, epoch_list):
    for obj in object_list:
        with open('param_config.yml', 'r') as f:
            config_file = yaml.safe_load(f)
        
        config_file['wandb_name'] = 'HeldOutObject-'+model+'-'+obj
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

        train_files = [x for x in os.listdir(input_dir) if obj not in x]
        test_files = [x for x in os.listdir(input_dir) if obj in x]

        for f_train in train_files:
            shutil.copyfile(input_dir+f_train, data_path+'train/'+f_train)

        for f_test in test_files:
            shutil.copyfile(input_dir+f_test, data_path+'test/'+f_test)

        print("Num train samples:", len(os.listdir(data_path+'train/')))
        print("Num test samples:", len(os.listdir(data_path+'test/')))

        trainer.main()

