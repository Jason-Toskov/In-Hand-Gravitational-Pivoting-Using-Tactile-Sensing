# References:
# https://stackoverflow.com/a/3114573

import copy
import time
import sys
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import wandb

from arg_set import parse_arguments
from util import SampleType, DataMode
from dataset import TactileDataset
from models import WindowMLP, RegressionLSTM

# Train one epoch
def train(device, loader, model, loss_func, optim, l1loss, data_mode):
    model.train()
    # loss for both pos and vel if both else only one
    loss_count = np.array([0.0,0.0]) if data_mode == DataMode.BOTH else 0
    abs_error_count = np.array([0.0,0.0]) if data_mode == DataMode.BOTH else 0

    for batch_num, (features, gt_angle, gt_velocity) in enumerate(loader):
        # gt is 'label' for only 1 target
        if data_mode == DataMode.POSITION:
            label = gt_angle
        elif data_mode == DataMode.VELOCITY:
            label = gt_velocity
        else:
            label = None
        
        out = model(features.to(device)).to(device)

        # Loss for one target
        if data_mode == DataMode.POSITION or data_mode == DataMode.VELOCITY:
            l1error = l1loss(out.squeeze(2), label.to(device))
            abs_error_count += l1error.item()

            loss = loss_func(out.squeeze(2), label.to(device))
            loss_count += loss.item()

            total_loss = loss + l1error

        # Loss for both pos and vel for both targets
        elif data_mode == DataMode.BOTH:
            vel_l1error = l1loss(out[:,:,0], gt_velocity.to(device))
            pos_l1error = l1loss(out[:,:,1], gt_angle.to(device))
            abs_error_count += [vel_l1error.item(), pos_l1error.item()]

            vel_loss = loss_func(out[:,:,0], gt_velocity.to(device))
            pos_loss = loss_func(out[:,:,1], gt_angle.to(device))
            loss_count += [vel_loss.item(), pos_loss.item()]

            total_loss = vel_l1error + pos_l1error + vel_loss + pos_loss

        # backprop
        optim.zero_grad()
        total_loss.backward()
        optim.step()
    
    loss_count /= batch_num+1
    abs_error_count /= batch_num+1

    return loss_count, abs_error_count

# Test one epoch
def test(device, loader, model, loss_func, optim, l1loss, data_mode, data):
    model.eval()

    is_both = data_mode == DataMode.BOTH
    # loss for both pos and vel if both else only one
    loss_count = np.array([0.0,0.0]) if data_mode == DataMode.BOTH else 0
    abs_error_count = np.array([0.0,0.0]) if data_mode == DataMode.BOTH else 0
    angle_error_running = np.zeros((2, 3)) if is_both else np.zeros(3)
    vel_error_running = np.zeros((2,3)) if is_both else np.zeros(3)

    with torch.no_grad():
        for batch_num, (features, gt_angle, gt_velocity) in enumerate(loader):
            
            # gt is 'label' for only 1 target
            if data_mode == DataMode.POSITION:
                label = gt_angle
            elif data_mode == DataMode.VELOCITY:
                label = gt_velocity
            else:
                label = None
            
            out = model(features.to(device)).to(device)

            # Loss for one target
            if data_mode == DataMode.POSITION or data_mode == DataMode.VELOCITY:
                l1error = l1loss(out.squeeze(2), label.to(device))
                abs_error_count += l1error.item()
                loss = loss_func(out.squeeze(2), label.to(device))
                loss_count += loss.item()

            # Loss for both pos and vel for both targets
            elif is_both:
                # For 'both', label is gt velocity and true_angle is gt angle
                vel_l1error = l1loss(out[:,:,0], gt_velocity.to(device))
                pos_l1error = l1loss(out[:,:,1], gt_angle.to(device))
                abs_error_count += [vel_l1error.item(), pos_l1error.item()]

                vel_loss = loss_func(out[:,:,0], gt_velocity.to(device))
                pos_loss = loss_func(out[:,:,1], gt_angle.to(device))
                loss_count += [vel_loss.item(), pos_loss.item()]

            if data_mode == DataMode.VELOCITY or is_both:
                # Get angle from integrating velocity
                vels =  out[:,:,0] if is_both else out
                integrated_vels = []
                for i, v in enumerate(vels.squeeze()):
                    if i:
                        integrated_vels.append(integrated_vels[i-1] + v/60)
                    else:
                        integrated_vels.append(v/60)
                
                integrated_vels = data.strechAngle(torch.tensor(integrated_vels), DataMode.VELOCITY)
                true_angle_stretched = data.strechAngle(gt_angle, DataMode.POSITION)

                # Get error for IS, DR and SS sections
                angle_error1 = l1loss(integrated_vels[:30], true_angle_stretched.squeeze()[:30]).item()
                angle_error2 = l1loss(integrated_vels[30:-60], true_angle_stretched.squeeze()[30:-60]).item()
                angle_error3 = l1loss(integrated_vels[-60:], true_angle_stretched.squeeze()[-60:]).item()

                # If DR section is empty set error to 0 (would be NaN otherwise)
                if len(integrated_vels[30:-60]) == 0:
                    print('Short sequence found!')
                    angle_error2 = 0.0

                angle_error = [angle_error1, angle_error2, angle_error3]

                if is_both:
                    angle_error_running[0] += angle_error
                else:
                    angle_error_running += angle_error

                # Absolute error in just velocity
                pred_vel =  out[:,:,0].squeeze() if is_both else out.squeeze()
                pred_vel = data.strechAngle(pred_vel, DataMode.VELOCITY)
                true_vel_stretched = data.strechAngle(gt_velocity.squeeze(), DataMode.VELOCITY)

                # Get error for IS, DR and SS sections
                vel_error1 = l1loss(pred_vel[:30], true_vel_stretched.to(device).squeeze()[:30]).item()
                vel_error2 = l1loss(pred_vel[30:-60], true_vel_stretched.to(device).squeeze()[30:-60]).item()
                vel_error3 = l1loss(pred_vel[-60:], true_vel_stretched.to(device).squeeze()[-60:]).item()

                # If DR section is empty set error to 0 (would be NaN otherwise)
                if len(pred_vel[30:-60]) == 0:
                    print('Short sequence found!')
                    vel_error2 = 0.0

                vel_error = [vel_error1, vel_error2, vel_error3]
                if is_both:
                    vel_error_running[1] += vel_error
                else:
                    vel_error_running += vel_error

                                
            if data_mode == DataMode.POSITION or is_both: 
                # Get velocity from differentiating angle
                angles = out[:,:,1] if is_both else out
                vel_from_diff = []
                angles = angles.squeeze()
                for i, a in enumerate(angles):
                    if i:
                        vel_from_diff.append((a - angles[i-1])*60)
                    else:
                        vel_from_diff.append((a-0)*60)

                vel_from_diff = data.strechAngle(torch.tensor(vel_from_diff), DataMode.POSITION)
                gt_vel_stretched = data.strechAngle(gt_velocity, DataMode.POSITION)

                # Get error for IS, DR and SS sections
                vel_error1 = l1loss(vel_from_diff[:30], gt_vel_stretched.squeeze()[:30]).item()
                vel_error2 = l1loss(vel_from_diff[30:-60], gt_vel_stretched.squeeze()[30:-60]).item()
                vel_error3 = l1loss(vel_from_diff[-60:], gt_vel_stretched.squeeze()[-60:]).item()

                # If DR section is empty set error to 0 (would be NaN otherwise)
                if len(vel_from_diff[30:-60]) == 0:
                    print('Short sequence found!')
                    vel_error2 = 0.0

                vel_error = [vel_error1, vel_error2, vel_error3]
                if is_both:
                    vel_error_running[0] += vel_error
                else:
                    vel_error_running += vel_error

                # absolute error just in position
                pred_angle = out[:,:,1].squeeze() if is_both else out.squeeze()

                pred_angle = data.strechAngle(pred_angle, DataMode.POSITION)
                true_angle_stretched = data.strechAngle(gt_angle, DataMode.POSITION)

                # Get error for IS, DR and SS sections
                angle_error1 = l1loss(pred_angle[:30], true_angle_stretched.to(device).squeeze()[:30]).item()
                angle_error2 = l1loss(pred_angle[30:-60], true_angle_stretched.to(device).squeeze()[30:-60]).item()
                angle_error3 = l1loss(pred_angle[-60:], true_angle_stretched.to(device).squeeze()[-60:]).item()

                # If DR section is empty set error to 0 (would be NaN otherwise)
                if len(pred_angle[30:-60]) == 0:
                    print('Short sequence found!')
                    angle_error2 = 0.0

                angle_error = [angle_error1, angle_error2, angle_error3]
                if is_both:
                    angle_error_running[1] += angle_error
                else:
                    angle_error_running += angle_error

        loss_count /= batch_num+1
        abs_error_count /= batch_num+1
        angle_error_running /= batch_num+1
        vel_error_running /= batch_num+1

    return loss_count, abs_error_count, angle_error_running, vel_error_running

# Get plot of a set number of sample sequences 
def plot_examples(subplt_shape, dataset, raw_dataset, dataloader, device, model, label_scale, plt_title, wandb_dir, data_mode, alt_mode=False, is_both = False):
    # Format is a matplotlib subplot with shape given by 'subplt_shape'
    fig, axs = plt.subplots(*subplt_shape)
    sentinel = object()

    # Check whether to integrate/differentiate
    integrate_vel = False
    diff_angle = False
    if data_mode == DataMode.POSITION:
        label_idx = 1
        diff_angle = True if alt_mode else False
    elif data_mode == DataMode.VELOCITY:
        label_idx = 2
        integrate_vel = True if alt_mode else False

    # Get examples from a single iterator
    dl_iter = iter(dataloader)

    for ax in axs.flat:
        # Make sure the iterator isnt empty
        # https://stackoverflow.com/a/3114573
        dl_out = next(dl_iter, sentinel)
        # If the iterator is empty don't continue looping
        if dl_out is sentinel:
            break
        count = 0
        # Check the plots aren't constant 0 and the iterator isn't done
        while(max(dl_out[label_idx].squeeze())-min(dl_out[label_idx].squeeze()) < 5/label_scale) and count < len(dataset):
            dl_out = next(dl_iter, sentinel)
            if dl_out is sentinel:
                break
            count += 1

        out = model(dl_out[0].to(device)).squeeze().to(device)

        if integrate_vel:
            # Get angle from integrating velocity
            vels =  out[:,0] if is_both else out
            integrated_vels = []
            for i, v in enumerate(vels):
                if i:
                    integrated_vels.append(integrated_vels[i-1] + v/60)
                else:
                    integrated_vels.append(v/60)
            integ_angle = torch.tensor(integrated_vels)

            gt = dl_out[1].squeeze().detach().to('cpu')
            pred = integ_angle.detach().to('cpu')

        elif diff_angle:
            # Get velocity from differentiating angle
            angles = out[:,1] if is_both else out
            angles = angles.squeeze()
            vel_from_diff = []
            for i, a in enumerate(angles):
                if i:
                    vel_from_diff.append((a - angles[i-1])*60)
                else:
                    vel_from_diff.append((a-0)*60)
            diffed_vel = torch.tensor(vel_from_diff)

            gt = dl_out[2].squeeze().detach().to('cpu')
            pred = diffed_vel.detach().to('cpu')

        else:
            if is_both:
                gt = dl_out[2].squeeze().detach().to('cpu') if data_mode == DataMode.VELOCITY else dl_out[1].squeeze().detach().to('cpu')
            else:
                gt = dl_out[label_idx].squeeze().detach().to('cpu')
            idx = 0 if data_mode == DataMode.VELOCITY else 1
            pred = out[:,idx].detach().to('cpu') if is_both else out.detach().to('cpu')

        if integrate_vel and is_both:
            gt_stretch = DataMode.POSITION
        elif diff_angle and is_both:
            gt_stretch = DataMode.VELOCITY
        else:
            gt_stretch = data_mode

        # get the x axis range
        range_len = len(out[:,0]) if is_both else len(out)
        x_range = [*range(range_len)]

        gt_plot = raw_dataset.strechAngle(gt, gt_stretch).tolist()
        pred_plot = raw_dataset.strechAngle(pred, data_mode).tolist()

        ax.plot(x_range, gt_plot, label='Ground truth')
        ax.plot(x_range, pred_plot, label='Prediction')

    fig.suptitle(plt_title)
    wandb.log({wandb_dir: fig})


def main():
    # Couple of params not too relevant to experiments in paper
    subplt_shape = (10,10)
    seq_length = None
    angle_difference = False

    # Parse args
    args = parse_arguments()

    # Check if sweeping or using default config
    if len(sys.argv) == 1:
        config_file = args.config
        with open(config_file, 'r') as f:
            cfg_input = yaml.safe_load(f)
    elif len(sys.argv) > 1:
        del args.config
        cfg_input = args
    else:
        raise ValueError("Weird arg error")

    # init wandb run
    run = wandb.init(project=cfg_input['wandb_project'],
                     entity="deep-tactile-rotatation-estimation",
                     config=cfg_input,
                     notes= cfg_input['wandb_run_notes'],
                     mode= None if cfg_input['use_wandb'] else 'disabled',
                     name = None if cfg_input['wandb_name'] == 'None' else cfg_input['wandb_name']
                    )

    config = wandb.config

    ### Use config prarms to set a bunch of internal params
    # Sample type
    if config['sample'] == 'random':
        sample_type = SampleType.RANDOM
    elif config['sample'] == 'center':
        sample_type = SampleType.CENTER
    elif config['sample'] == 'front':
        sample_type = SampleType.FRONT

    # Data mode
    if config['model_prediction'] == 'position':
        data_mode = DataMode.POSITION
        label_scale = config['label_scale_position']
    elif config['model_prediction'] == 'velocity':
        data_mode = DataMode.VELOCITY
        label_scale = config['label_scale_velocity']
    elif config['model_prediction'] == 'both':
        data_mode = DataMode.BOTH
        label_scale = [config['label_scale_velocity'], config['label_scale_position']]

    # Model type
    if config['model_type'] == 'MLP':
        model_type = 'WindowMLP'
    elif config['model_type'] == 'LSTM':
        model_type = 'RegressionLSTM'

    # If testing only loop once with a batch size of 1
    # this is because the test finction will only work with batch size = 1
    if config['test_only']:
        train_batch_size = 1
        num_epochs = 1
    else:
        train_batch_size = config['train_batch_size']
        num_epochs = config["num_epochs"]
    
    print(config)
    print("Using GPU:", torch.cuda.is_available())

    # Set device to GPU_indx if GPU is avaliable
    GPU_indx = 0
    device = torch.device(GPU_indx if torch.cuda.is_available() else 'cpu')

    # Create dataset/dataloaders
    if config['manual_test_set']:
        data = TactileDataset(config["data_path"], label_scale=label_scale, sample_type=sample_type, seq_length=seq_length,
                            angle_difference=angle_difference, num_features=config['num_features'], transform=None, mode='train', data_mode=data_mode)
        train_data = TactileDataset(config["data_path"], label_scale=label_scale, sample_type=sample_type, seq_length=seq_length,
                            angle_difference=angle_difference, num_features=config['num_features'], transform=None, mode='train', data_mode=data_mode)
        test_data = TactileDataset(config["data_path"], label_scale=label_scale, sample_type=sample_type, seq_length=seq_length,
                            angle_difference=angle_difference, num_features=config['num_features'], transform=None, mode='test', data_mode=data_mode)
    else:
        data = TactileDataset(config["data_path"], label_scale=label_scale, sample_type=sample_type, seq_length=seq_length,
                            angle_difference=angle_difference, num_features=config['num_features'], transform=None, data_mode=data_mode)
        
        # Take a random split with fraction defined by config["train_frac"]
        train_data_length = round(len(data)*config["train_frac"])
        test_data_length = len(data) - train_data_length
        train_data, test_data = random_split(
            data, [train_data_length, test_data_length], generator=torch.Generator().manual_seed(42))

    # Create data loaders
    train_loader = DataLoader(
        train_data, batch_size=train_batch_size, shuffle=True, collate_fn=data.collate_fn)
    test_loader = DataLoader(
        test_data, batch_size=config["test_batch_size"], shuffle=False, collate_fn=data.collate_fn)

    # Create model
    model = eval(model_type+'''(
        device, config['num_features'], config["hidden_size"], config["num_layers"], 
        config["dropout"], config["MLP_window"], data_mode
    )''')

    # Load model if config["resume_from_checkpoint"]
    if config["resume_from_checkpoint"]:
        model.load_state_dict(torch.load(config["model_path"]))
        print('Model loaded from:',config["model_path"])

    model = model.to(device)

    # Define loss and optimization functions
    loss_func = nn.MSELoss()
    l1loss = nn.L1Loss()

    optim = torch.optim.Adam(model.parameters(
    ), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    # Run training
    best_model = copy.deepcopy(model)
    lowest_error = 1e5
    old_time = time.time()
    best_epoch = -1
    # Train loop
    for i in range(num_epochs):
        # test() if test_only else train()
        if config['test_only']:
            loss_train, abs_error_train, angle_error_train, vel_error_train = test(
                device, train_loader, model, loss_func, optim, l1loss, data_mode, data)
        else:
            loss_train, abs_error_train = train(
                device, train_loader, model, loss_func, optim, l1loss, data_mode)

        loss_test, abs_error_test, angle_error_test, vel_error_test = test(
            device, test_loader, model, loss_func, optim, l1loss, data_mode, data)

        if data_mode == DataMode.BOTH:
            total_test_loss = loss_test.sum()
            # For 'both' take position error as the error to track
            total_test_error = data.strechAngle(abs_error_test[1], DataMode.POSITION)
            # Log a bunch of errors to wandb
            wandb.log({
                "Loss/Velocity_train": loss_train[0],
                "Loss/Velocity_test": loss_test[0],
                "Loss/Position_train": loss_train[1],
                "Loss/Position_test": loss_test[1],
                "abs_error/Velocity_train": data.strechAngle(abs_error_train[0], DataMode.VELOCITY),
                "abs_error/Velocity_test": data.strechAngle(abs_error_test[0], DataMode.VELOCITY),
                "abs_error/Position_train": data.strechAngle(abs_error_train[1], DataMode.POSITION),
                "abs_error/Position_test": data.strechAngle(abs_error_test[1], DataMode.POSITION),
                "start_error/velocity_integrated_angle_test": angle_error_test[0,0],
                "middle_error/velocity_integrated_angle_test": angle_error_test[0,1],
                "end_error/velocity_integrated_angle_test": angle_error_test[0,2],
                "start_error/raw_angle_prediction_test": angle_error_test[1,0],
                "middle_error/raw_angle_prediction_test": angle_error_test[1,1],
                "end_error/raw_angle_prediction_test": angle_error_test[1,2],
                "start_error/angle_diffed_velocity_test": vel_error_test[0,0],
                "middle_error/angle_diffed_velocity_test": vel_error_test[0,1],
                "end_error/angle_diffed_velocity_test": vel_error_test[0,2],
                "start_error/raw_vel_prediction_test": vel_error_test[1,0],
                "middle_error/raw_vel_prediction_test": vel_error_test[1,1],
                "end_error/raw_vel_prediction_test": vel_error_test[1,2],
            })
            if config['test_only']:
                # If test_only log the errors for the 3 sections for the train dataset as well
                wandb.log({
                    "start_error/velocity_integrated_angle_train": angle_error_train[0,0],
                    "middle_error/velocity_integrated_angle_train": angle_error_train[0,1],
                    "end_error/velocity_integrated_angle_train": angle_error_train[0,2],
                    "start_error/raw_angle_prediction_train": angle_error_train[1,0],
                    "middle_error/raw_angle_prediction_train": angle_error_train[1,1],
                    "end_error/raw_angle_prediction_train": angle_error_train[1,2],
                    "start_error/angle_diffed_velocity_train": vel_error_train[0,0],
                    "middle_error/angle_diffed_velocity_train": vel_error_train[0,1],
                    "end_error/angle_diffed_velocity_train": vel_error_train[0,2],
                    "start_error/raw_vel_prediction_train": vel_error_train[1,0],
                    "middle_error/raw_vel_prediction_train": vel_error_train[1,1],
                    "end_error/raw_vel_prediction_train": vel_error_train[1,2],
                })
        else:
            if data_mode == DataMode.VELOCITY:
                data_type = 'Velocity'
                angle_section_error = 'velocity_integrated_angle'
                velocity_section_error = 'raw_vel_prediction'
            else:
                data_type = 'Position'
                angle_section_error = 'raw_angle_prediction'
                velocity_section_error = 'angle_diffed_velocity'

            total_test_error = data.strechAngle(abs_error_test)
            total_test_loss = loss_test
            wandb.log({
                "Loss/"+data_type+"_train": loss_train,
                "Loss/"+data_type+"_test": loss_test,
                "abs_error/"+data_type+"_train": data.strechAngle(abs_error_train),
                "abs_error/"+data_type+"_test": data.strechAngle(abs_error_test),
                "start_error/"+angle_section_error+"_test": angle_error_test[0],
                "middle_error/"+angle_section_error+"_test": angle_error_test[1],
                "end_error/"+angle_section_error+"_test": angle_error_test[2],
                "start_error/"+velocity_section_error+"_test": vel_error_test[0],
                "middle_error/"+velocity_section_error+"_test": vel_error_test[1],
                "end_error/"+velocity_section_error+"_test": vel_error_test[2],
            })
            if config['test_only']:
                wandb.log({
                    "start_error/"+angle_section_error+"_train": angle_error_train[0],
                    "middle_error/"+angle_section_error+"_train": angle_error_train[1],
                    "end_error/"+angle_section_error+"_train": angle_error_train[2],
                    "start_error/"+velocity_section_error+"_train": vel_error_train[0],
                    "middle_error/"+velocity_section_error+"_train": vel_error_train[1],
                    "end_error/"+velocity_section_error+"_train": vel_error_train[2],
                })

        new_time = time.time()
        print("Epoch: %i, Test error: %f, Test loss: %f, Time taken: %.2f sec/epoch" %
                (i, total_test_error, total_test_loss, new_time-old_time))
        old_time = copy.deepcopy(new_time)
        
        # Save the best model
        if total_test_error < lowest_error:
            best_epoch = i
            best_model_dict = copy.deepcopy(model.state_dict())
            torch.save(best_model_dict, config["model_path"])
            lowest_error = total_test_error
            print("new best!")

    print("Lowest error was: %f" % (lowest_error))
    print("Lowest error occured on epoch",best_epoch)

    # Log the best model to wandb
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(config["model_path"])
    run.log_artifact(artifact)

    # Plot the sample sequences
    if not angle_difference and config['use_wandb']:

        # Load best model
        model = eval(model_type+'''(
            device, config['num_features'], config["hidden_size"], config["num_layers"], 
            config["dropout"], config["MLP_window"], data_mode
        )''')
        best_model.load_state_dict(torch.load(config["model_path"]))
        best_model = best_model.to(device)

        # Refresh loaders
        train_loader = DataLoader(
            train_data, batch_size=1, shuffle=True, collate_fn=data.collate_fn)
        test_loader = DataLoader(
            test_data, batch_size=1, shuffle=False, collate_fn=data.collate_fn)

        # Plot examples here 
        if data_mode == DataMode.POSITION:
            plot_examples(subplt_shape, train_data, data, train_loader, device, best_model, label_scale,
                          'Train examples (position prediction)', 'Position_prediction_examples/Train', DataMode.POSITION)
            plot_examples(subplt_shape, test_data, data, test_loader, device, best_model, label_scale,
                          'Test examples (position prediction)', 'Position_prediction_examples/Test', DataMode.POSITION)
            plot_examples(subplt_shape, train_data, data, train_loader, device, best_model, label_scale,
                          'Train examples (diffed velocity)', 'Diffed_velocity_examples/Train', DataMode.POSITION, alt_mode=True)
            plot_examples(subplt_shape, test_data, data, test_loader, device, best_model, label_scale,
                          'Test examples (diffed velocity)', 'Diffed_velocity_examples/Test', DataMode.POSITION, alt_mode=True)
        elif data_mode == DataMode.VELOCITY:
            # Note the 'POSITION' data type here will plot the raw velocity
            plot_examples(subplt_shape, train_data, data, train_loader, device, best_model, label_scale,
                          'Train examples (velocity prediction)', 'Velocity_prediction_examples/Train', DataMode.VELOCITY)
            plot_examples(subplt_shape, test_data, data, test_loader, device, best_model, label_scale,
                          'Test examples (velocity prediction)', 'Velocity_prediction_examples/Test', DataMode.VELOCITY)
            plot_examples(subplt_shape, train_data, data, train_loader, device, best_model, label_scale,
                          'Train examples (integrated angle)', 'Integrated_angle_examples/Train', DataMode.VELOCITY, alt_mode=True)
            plot_examples(subplt_shape, test_data, data, test_loader, device, best_model, label_scale,
                          'Test examples (integrated angle)', 'Integrated_angle_examples/Test', DataMode.VELOCITY, alt_mode=True)
        elif data_mode == DataMode.BOTH:
            plot_examples(subplt_shape, train_data, data, train_loader, device, best_model, label_scale[1],
                          'Train examples (velocity prediction)', 'Velocity_prediction_examples/Train', DataMode.VELOCITY, alt_mode=False, is_both=True)
            plot_examples(subplt_shape, test_data, data, test_loader, device, best_model, label_scale[1],
                          'Test examples (velocity prediction)', 'Velocity_prediction_examples/Test', DataMode.VELOCITY, alt_mode=False, is_both=True)
            plot_examples(subplt_shape, train_data, data, train_loader, device, best_model, label_scale[1],
                          'Train examples (integrated angle)', 'Integrated_angle_examples/Train', DataMode.VELOCITY, alt_mode=True, is_both=True)
            plot_examples(subplt_shape, test_data, data, test_loader, device, best_model, label_scale[1],
                          'Test examples (integrated angle)', 'Integrated_angle_examples/Test', DataMode.VELOCITY, alt_mode=True, is_both=True)
            plot_examples(subplt_shape, train_data, data, train_loader, device, best_model, label_scale[0],
                          'Train examples (position prediction)', 'Position_prediction_examples/Train', DataMode.POSITION, alt_mode=False, is_both=True)
            plot_examples(subplt_shape, test_data, data, test_loader, device, best_model, label_scale[0],
                          'Test examples (position prediction)', 'Position_prediction_examples/Test', DataMode.POSITION, alt_mode=False, is_both=True)
            plot_examples(subplt_shape, train_data, data, train_loader, device, best_model, label_scale[0],
                          'Train examples (diffed velocity)', 'Diffed_velocity_examples/Train', DataMode.POSITION, alt_mode=True, is_both=True)
            plot_examples(subplt_shape, test_data, data, test_loader, device, best_model, label_scale[0],
                          'Test examples (diffed velocity)', 'Diffed_velocity_examples/Test', DataMode.POSITION, alt_mode=True, is_both=True)

    run.finish()
    print("Training complete!")

if __name__ == "__main__":
    main()
