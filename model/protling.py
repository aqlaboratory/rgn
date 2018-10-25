#!/usr/bin/python

# imports
import os
import sys
import signal
import argparse
import fileinput
import numpy as np
import tensorflow as tf
from utils import *
from copy import deepcopy
from glob import glob
from shutil import rmtree
from pprint import pprint
from setproctitle import setproctitle
from model import RGNModel
from config import RGNConfig, RunConfig

# constant directory and file names
RUNS_DIRNAME = 'runs'
DATAS_DIRNAME = 'data'
CHECKPOINTS_DIRNAME = 'checkpoints'
LOGS_DIRNAME = 'logs'
ALPHABETS_DIRNAME = 'alphabets'
FULL_TRAINING_DIRNAME     = 'training'
SAMPLE_VALIDATION_DIRNAME = 'validation'
FULL_TESTING_DIRNAME      = 'testing'
TRAINING_OUTPUTS_DIRNAME   = 'outputsTraining'
VALIDATION_OUTPUTS_DIRNAME = 'outputsValidation'
TESTING_OUTPUTS_DIRNAME    = 'outputsTesting'

# exception classes
class MilestoneError(RuntimeError):
    """ Exception raised for missing milestone """
    pass

class DeadGradientError(RuntimeError):
    """ Exception raised for zero gradient """
    pass

# logging functions
def evaluate_and_log(log_file, configs, models, session):
    # evaluation of weighted losses
    wt_train_loss_dict = models['eval_wt_train'].evaluate(session) if configs['run'].evaluation['include_weighted_training']   else {}
    wt_val_loss_dict   = models['eval_wt_val'].evaluate(session)   if configs['run'].evaluation['include_weighted_validation'] else {}
    wt_test_loss_dict  = models['eval_wt_test'].evaluate(session)  if configs['run'].evaluation['include_weighted_testing']    else {}

    # diagnostics
    if configs['run'].evaluation['include_diagnostics']: 
        diagnostics = models['training'].diagnose(session)
    else:
        diagnostics = {k: float('nan') for k in ('min_weight', 'max_weight', 'min_grad', 'max_grad', 
                                                 'curriculum_step', 'curriculum_quantiles')}

    # Retrieve the correct loss.
    for loss_key in ['tertiary_loss_all']:
        if loss_key in wt_train_loss_dict:
            wt_train_loss = wt_train_loss_dict[loss_key]
            break
    else:
        wt_train_loss = float('nan')

    if configs['run'].evaluation['include_weighted_validation']:
        wt_val_loss = {}
        for loss_type in ['tertiary_loss', 'min_tertiary_loss_achieved']:
            for subgroup in ['all'] + configs['eval_wt_val'].io['evaluation_sub_groups']:
                loss_key = loss_type + '_' + subgroup
                wt_val_loss.update({loss_key: wt_val_loss_dict.get(loss_key, float('nan'))})
        wt_val_loss_subgroups_string = ''.join(map(lambda grp: '\tValidation_' + grp + ': {tertiary_loss_' + grp + ':.3f}',
                                                   configs['eval_wt_val'].io['evaluation_sub_groups']))
    else:
        wt_val_loss = {'tertiary_loss_all': float('nan')}
        wt_val_loss_subgroups_string = '' 

    for loss_key in ['tertiary_loss_all']:
        if loss_key in wt_test_loss_dict:
            wt_test_loss  = wt_test_loss_dict[loss_key]
            break
    else:
        wt_test_loss = float('nan')

    # Log string
    global_step = models['training'].current_step(session)
    base_log = ('Iteration: {0}\tTrain: {1:.3f}\t' + \
                'Validation: {2:.3f}\tTest: {3:.3f}\t' + \
                'Weight: {min_weight:.4e} {max_weight:.4e}\t' + \
                'Update: {min_grad:.4e} {max_grad:.4e}' + \
                wt_val_loss_subgroups_string
               ).format(global_step, wt_train_loss, wt_val_loss['tertiary_loss_all'], wt_test_loss, **merge_dicts(diagnostics, wt_val_loss))
    
    # Additional diagnostics and losses if there's a curriculum.
    if configs['training'].curriculum['mode'] is not None:
        # evaluation of unweighted losses
        unwt_train_loss_dict = models['eval_unwt_train'].evaluate(session) if configs['run'].evaluation['include_unweighted_training']   else {}
        unwt_val_loss_dict   = models['eval_unwt_val'].evaluate(session)   if configs['run'].evaluation['include_unweighted_validation'] else {}
        unwt_test_loss_dict  = models['eval_unwt_test'].evaluate(session)  if configs['run'].evaluation['include_unweighted_testing']    else {}

        # Retrieve the correct loss.
        for loss_key in ['tertiary_loss_all']:
            if loss_key in unwt_train_loss_dict:
                unwt_train_loss = unwt_train_loss_dict[loss_key]
                break
        else:
            unwt_train_loss = float('nan')

        if configs['run'].evaluation['include_unweighted_validation']:
            unwt_val_loss = {}
            for loss_type in ['tertiary_loss', 'min_tertiary_loss_achieved']:
                for subgroup in ['all'] + configs['eval_unwt_val'].io['evaluation_sub_groups']:
                    loss_key = loss_type + '_' + subgroup
                    unwt_val_loss.update({loss_key: unwt_val_loss_dict.get(loss_key, float('nan'))})
            unwt_val_loss_subgroups_string = ''.join(map(lambda grp: '\tUnweighted Validation_' + grp + ': {tertiary_loss_' + grp + ':.3f}', 
                                                         configs['eval_unwt_val'].io['evaluation_sub_groups']))
        else:
            unwt_val_loss = {'tertiary_loss_all': float('nan')}
            unwt_val_loss_subgroups_string = ''

        for loss_key in ['tertiary_loss_all']:
            if loss_key in unwt_test_loss_dict:
                unwt_test_loss   = unwt_test_loss_dict[loss_key]
                break
        else:
            unwt_test_loss   = float('nan')

        # Log string
        extended_log = ('\tCurriculum Iteration: {curriculum_step:.3f}\t' + \
                        'Unweighted Train: {0:.3f}\t' + \
                        'Unweighted Validation: {1:.3f}\t' + \
                        'Unweighted Test: {2:.3f}\t' + \
                        'Curriculum Quantile: {curriculum_quantiles}' + \
                        unwt_val_loss_subgroups_string
                       ).format(unwt_train_loss, unwt_val_loss['tertiary_loss_all'], unwt_test_loss, **merge_dicts(diagnostics, unwt_val_loss))
    else:
        extended_log = ''

    # Log to disk
    with open(log_file, 'a') as f: f.write(base_log + extended_log + '\n')

    if 'alphabet' in diagnostics: 
        with open(log_file + '.alphabet', 'a') as f:
            np.savetxt(f, diagnostics['alphabet'], footer='\n')

    # prep return 'package'
    diagnostics.update({      'wt_train_loss':   wt_train_loss,   'wt_val_loss':   wt_val_loss,   'wt_test_loss':   wt_test_loss})
    if configs['training'].curriculum['mode'] is not None:
        diagnostics.update({'unwt_train_loss': unwt_train_loss, 'unwt_val_loss': unwt_val_loss, 'unwt_test_loss': unwt_test_loss})

    return diagnostics

def predict_and_log(log_dir, configs, models, session):
    # assumes that the validation reference designation (wt vs. unwt) can be used for the training and test sets as well
    val_ref_set_prefix = 'un' if configs['run'].optimization['validation_reference'] == 'unweighted' else ''

    for label, model in models.iteritems():
        if 'eval' in label:
            generate = True

            for case in switch(label):
                if case('eval_' + val_ref_set_prefix + 'wt_train'):
                    outputs_dir = os.path.join(log_dir, TRAINING_OUTPUTS_DIRNAME)
                elif case('eval_' + val_ref_set_prefix + 'wt_val'):
                    outputs_dir = os.path.join(log_dir, VALIDATION_OUTPUTS_DIRNAME)
                elif case('eval_' + val_ref_set_prefix + 'wt_test'):
                    outputs_dir = os.path.join(log_dir, TESTING_OUTPUTS_DIRNAME)
                else:
                    generate = False

            if generate:
                if not os.path.exists(outputs_dir): os.makedirs(outputs_dir)

                for _ in range(configs[label].queueing['num_evaluation_invocations']):
                    dicts = model.predict(session)
                    for idx, dict_ in dicts.iteritems():
                        if 'tertiary'  in dict_:
                            np.savetxt(os.path.join(outputs_dir, idx + '.tertiary'), dict_['tertiary'], header='\n')
                        if 'recurrent_states' in dict_:
                            np.savetxt(os.path.join(outputs_dir, idx + '.recurrent_states'), dict_['recurrent_states'])

def loop(args):
    # create config and model collection objects, and retrieve the run config
    configs = {}
    models  = {}
    configs.update({'run': RunConfig(args.config_file)})

    # set GPU-related environmental options and config settings
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu) if args.gpu is not None else ''
    setproctitle('RGN ' + configs['run'].names['run'] + ' on ' + os.getenv('CUDA_VISIBLE_DEVICES', 'CPU'))

    # derived files and directories
    base_dir        = args.base_directory
    run_dir         = os.path.join(base_dir, RUNS_DIRNAME,        configs['run'].names['run'], configs['run'].names['dataset'])
    data_dir        = os.path.join(base_dir, DATAS_DIRNAME,       configs['run'].names['dataset'])
    checkpoints_dir = os.path.join(run_dir,  CHECKPOINTS_DIRNAME, '')
    logs_dir        = os.path.join(run_dir,  LOGS_DIRNAME,        '')
    stdout_err_file = os.path.join(base_dir, LOGS_DIRNAME,        configs['run'].names['run'] + '.log')
    alphabet_file   = os.path.join(data_dir, ALPHABETS_DIRNAME,   configs['run'].names['alphabet'] + '.csv') if configs['run'].names['alphabet'] is not None else None

    # this is all for evaluation models (including training, so training_batch_size is for evaluation)
    full_training_glob     = os.path.join(data_dir, FULL_TRAINING_DIRNAME,     configs['run'].io['full_training_glob'])
    sample_training_glob   = os.path.join(data_dir, FULL_TRAINING_DIRNAME,   configs['run'].io['sample_training_glob'])
    training_batch_size    = configs['run'].evaluation['num_training_samples']
    training_invocations   = configs['run'].evaluation['num_training_invocations']

    validation_glob        = os.path.join(data_dir, SAMPLE_VALIDATION_DIRNAME, configs['run'].io['sample_validation_glob'])
    validation_batch_size  = configs['run'].evaluation['num_validation_samples']
    validation_invocations = configs['run'].evaluation['num_validation_invocations']

    testing_glob           = os.path.join(data_dir, FULL_TESTING_DIRNAME,      configs['run'].io['full_testing_glob'])
    testing_batch_size     = configs['run'].evaluation['num_testing_samples']
    testing_invocations    = configs['run'].evaluation['num_testing_invocations']

    if not args.prediction_only:
        eval_num_epochs = None
    else:
        eval_num_epochs = 1
        training_batch_size = validation_batch_size = testing_batch_size = 1
        training_invocations = validation_invocations = testing_invocations = 1

    # redirect stdout/err to file
    sys.stderr.flush()
    if not os.path.exists(os.path.dirname(stdout_err_file)): os.makedirs(os.path.dirname(stdout_err_file))
    stdout_err_file_handle = open(stdout_err_file, 'w')
    os.dup2(stdout_err_file_handle.fileno(), sys.stderr.fileno())
    sys.stdout = stdout_err_file_handle

    # select device placement taking into consideration the interaction between training and evaluation models
    if configs['run'].computing['training_device'] == 'GPU' and configs['run'].computing['evaluation_device'] == 'GPU':
        fod_training   = {'/cpu:0': ['point_to_coordinate']}
        fod_evaluation = {'/cpu:0': ['point_to_coordinate']}
        dd_training   = ''
        dd_evaluation = ''
    elif configs['run'].computing['training_device'] == 'GPU' and configs['run'].computing['evaluation_device'] == 'CPU':
        fod_training   = {'/cpu:0': ['point_to_coordinate', 'loss_history']}
        fod_evaluation = {}
        dd_training   = ''
        dd_evaluation = '/cpu:0'
    else:
        fod_training   = {}
        fod_evaluation = {}
        dd_training   = '/cpu:0'
        dd_evaluation = '/cpu:0'

    # create models configuration templates
    configs.update({'training': RGNConfig(args.config_file, 
                                          {'name':                        'training',
                                           'dataFilesGlob':               full_training_glob,
                                           'checkpointsDirectory':        checkpoints_dir,
                                           'logsDirectory':               logs_dir,
                                           'fileQueueCapacity':           configs['run'].queueing['training_file_queue_capacity'],
                                           'batchQueueCapacity':          configs['run'].queueing['training_batch_queue_capacity'],
                                           'minAfterDequeue':             configs['run'].queueing['training_min_after_dequeue'],
                                           'shuffle':                     configs['run'].queueing['training_shuffle'],
                                           'tertiaryNormalization':       configs['run'].loss['training_tertiary_normalization'],
                                           'batchDependentNormalization': configs['run'].loss['training_batch_dependent_normalization'],
                                           'alphabetFile':                alphabet_file,
                                           'functionsOnDevices':          fod_training,
                                           'defaultDevice':               dd_training,
                                           'fillGPU':                     args.fill_gpu})})

    configs.update({'evaluation': RGNConfig(args.config_file, 
                                            {'fileQueueCapacity':           configs['run'].queueing['evaluation_file_queue_capacity'],
                                             'batchQueueCapacity':          configs['run'].queueing['evaluation_batch_queue_capacity'],
                                             'minAfterDequeue':             configs['run'].queueing['evaluation_min_after_dequeue'],
                                             'shuffle':                     configs['run'].queueing['evaluation_shuffle'],
                                             'tertiaryNormalization':       configs['run'].loss['evaluation_tertiary_normalization'],
                                             'batchDependentNormalization': configs['run'].loss['evaluation_batch_dependent_normalization'],
                                             'alphabetFile':                alphabet_file,
                                             'functionsOnDevices':          fod_evaluation,
                                             'defaultDevice':               dd_evaluation,
                                             'numEpochs':                   eval_num_epochs,
                                             'bucketBoundaries':            None})})

    # Override included evaluation models with list from command-line if specified (assumes none are included and then includes ones that are specified)
    if args.evaluation_model:
        for prefix in ['', 'un']:
            for group in ['training', 'validation', 'testing']:
                configs['run'].evaluation.update({'include_' + prefix + 'weighted_' + group: False})
        for entry in args.evaluation_model:
            configs['run'].evaluation.update({'include_' + entry: True})

    # Override other command-lind arguments
    if args.gpu_fraction: configs['training'].computing.update({'gpu_fraction': args.gpu_fraction})
    if args.milestone: configs['run'].optimization.update({'validation_milestone': dict(args.milestone)})

    # Ensure that correct validation reference is chosen if not predicting, and turn off evaluation loss if predicting
    if not args.prediction_only:
        if ((not configs['run'].evaluation['include_weighted_validation'])   and configs['run'].optimization['validation_reference'] == 'weighted') or \
           ((not configs['run'].evaluation['include_unweighted_validation']) and configs['run'].optimization['validation_reference'] == 'unweighted'):
            raise RuntimeError('Chosen validation reference is not included in run.')
    else:
        configs['evaluation'].loss['include'] = False

    # rescaling needed to adjust for how frequently loss_history is updated
    if configs['training'].curriculum['behavior'] == 'loss_change': 
        configs['training'].curriculum[  'change_num_iterations'] //= configs['run'].io['evaluation_frequency'] # result must be >=1
        configs['evaluation'].curriculum['change_num_iterations'] //= configs['run'].io['evaluation_frequency'] # ditto

    # create training model
    models = {}
    models.update({'training': RGNModel('training', configs['training'])})
    print('*** training configuration ***')
    pprint(configs['training'].__dict__)

    # create weighted training evaluation model (conditional)
    if configs['run'].evaluation['include_weighted_training']:
        configs.update({'eval_wt_train': deepcopy(configs['evaluation'])})
        configs['eval_wt_train'].io['name'] = 'evaluation_wt_training'
        configs['eval_wt_train'].io['data_files_glob'] = sample_training_glob
        configs['eval_wt_train'].optimization['batch_size'] = training_batch_size
        configs['eval_wt_train'].queueing['num_evaluation_invocations'] = training_invocations
        models.update({'eval_wt_train': RGNModel('evaluation', configs['eval_wt_train'])})
        print('\n\n\n*** weighted training evaluation configuration ***')
        pprint(configs['eval_wt_train'].__dict__)

    # create weighted validation evaluation model (conditional)
    if configs['run'].evaluation['include_weighted_validation']:
        configs.update({'eval_wt_val': deepcopy(configs['evaluation'])})
        configs['eval_wt_val'].io['name'] = 'evaluation_wt_validation'
        configs['eval_wt_val'].io['data_files_glob'] = validation_glob
        configs['eval_wt_val'].optimization['batch_size'] = validation_batch_size
        configs['eval_wt_val'].queueing['num_evaluation_invocations'] = validation_invocations
        if configs['run'].optimization['validation_reference'] == 'weighted': 
            configs['eval_wt_val'].curriculum['update_loss_history'] = True
        models.update({'eval_wt_val': RGNModel('evaluation', configs['eval_wt_val'])})
        print('\n\n\n*** weighted validation evaluation configuration ***')
        pprint(configs['eval_wt_val'].__dict__)

    # create weighted testing evaluation model (conditional)
    if configs['run'].evaluation['include_weighted_testing']:
        configs.update({'eval_wt_test': deepcopy(configs['evaluation'])})
        configs['eval_wt_test'].io['name'] = 'evaluation_wt_testing'
        configs['eval_wt_test'].io['data_files_glob'] = testing_glob
        configs['eval_wt_test'].optimization['batch_size'] = testing_batch_size
        configs['eval_wt_test'].queueing['num_evaluation_invocations'] = testing_invocations
        models.update({'eval_wt_test': RGNModel('evaluation', configs['eval_wt_test'])})
        print('\n\n\n*** weighted testing evaluation configuration ***')
        pprint(configs['eval_wt_test'].__dict__)

    # create equivalents for unweighted loss if there's a curriculum.
    if configs['training'].curriculum['mode'] is not None:
        # create unweighted training evaluation model (conditional)
        if configs['run'].evaluation['include_unweighted_training']:
            configs.update({'eval_unwt_train': deepcopy(configs['evaluation'])})
            configs['eval_unwt_train'].io['name'] = 'evaluation_unwt_training'
            configs['eval_unwt_train'].io['data_files_glob'] = sample_training_glob
            configs['eval_unwt_train'].optimization['batch_size'] = training_batch_size
            configs['eval_unwt_train'].queueing['num_evaluation_invocations'] = training_invocations
            configs['eval_unwt_train'].curriculum['mode'] = None
            configs['eval_unwt_train'].curriculum['behavior'] = None
            models.update({'eval_unwt_train': RGNModel('evaluation', configs['eval_unwt_train'])})
        
        # create unweighted validation evaluation model (conditional)
        if configs['run'].evaluation['include_unweighted_validation']:
            configs.update({'eval_unwt_val': deepcopy(configs['evaluation'])})
            configs['eval_unwt_val'].io['name'] = 'evaluation_unwt_validation'
            configs['eval_unwt_val'].io['data_files_glob'] = validation_glob
            configs['eval_unwt_val'].optimization['batch_size'] = validation_batch_size
            configs['eval_unwt_val'].queueing['num_evaluation_invocations'] = validation_invocations
            configs['eval_unwt_val'].curriculum['mode'] = None
            configs['eval_unwt_val'].curriculum['behavior'] = None
            if configs['run'].optimization['validation_reference'] == 'unweighted': 
                configs['eval_unwt_val'].curriculum['update_loss_history'] = True
            models.update({'eval_unwt_val': RGNModel('evaluation', configs['eval_unwt_val'])})

        # create unweighted testing evaluation model (conditional)
        if configs['run'].evaluation['include_unweighted_testing']:
            configs.update({'eval_unwt_test': deepcopy(configs['evaluation'])})
            configs['eval_unwt_test'].io['name'] = 'evaluation_unwt_testing'
            configs['eval_unwt_test'].io['data_files_glob'] = testing_glob
            configs['eval_unwt_test'].optimization['batch_size'] = testing_batch_size
            configs['eval_unwt_test'].queueing['num_evaluation_invocations'] = testing_invocations
            configs['eval_unwt_test'].curriculum['mode'] = None
            configs['eval_unwt_test'].curriculum['behavior'] = None
            models.update({'eval_unwt_test': RGNModel('evaluation', configs['eval_unwt_test'])})

    # start head model and related prep
    stdout_err_file_handle.flush()
    session = models['training'].start(models.values())
    global_step = models['training'].current_step(session)
    current_log_step = (global_step // configs['run'].io['prediction_frequency']) + 1
    log_dir = os.path.join(run_dir, str(current_log_step))
    restart = False

    # predict or train depending on set mode behavior
    if args.prediction_only:
        try:
            while not models['training'].is_done():
                predict_and_log(log_dir, configs, models, session)
        except tf.errors.OutOfRangeError:
            pass
        except:
            print('Unexpected error: ', sys.exc_info()[0])
            raise
        finally:
            if models['training']._is_started: models['training'].finish(session, save=False)
            stdout_err_file_handle.close()
    else:
        # clean up post last checkpoint residue if any
        if global_step != 0:
            # remove future directories
            last_log_step = sorted([int(os.path.basename(os.path.normpath(dir))) for dir in glob(os.path.join(run_dir, '*[0-9]'))])[-1]
            for step in range(current_log_step + 1, last_log_step + 1): rmtree(os.path.join(run_dir, str(step))) 

            # remove future log entries in current log files
            log_file = os.path.join(log_dir, 'error.log')
            if os.path.exists(log_file):
                with open(log_file, 'rw+') as f:
                    while True:
                        new_line = f.readline().split()
                        if len(new_line) > 1:
                            step = int(new_line[1])
                            if step == global_step:
                                f.truncate()
                                break
                        else: # reached end without seeing global_step, means checkpoint is ahead of last recorded log entry
                            break

        # training loop
        try:
            while not models['training'].is_done():
                # Train for one step
                global_step, ids = models['training'].train(session)

                # Set and create logging directory and files if needed
                log_dir = os.path.join(run_dir, str((global_step // configs['run'].io['prediction_frequency']) + 1))
                log_file = os.path.join(log_dir, 'error.log')
                if not os.path.exists(log_dir): os.makedirs(log_dir)

                # Evaluate error, get diagnostics, and raise exceptions if necessary
                if global_step % configs['run'].io['evaluation_frequency'] == 0:
                    diagnostics = evaluate_and_log(log_file, configs, models, session)

                    # restart if a milestone is missed
                    val_ref_set_prefix = 'un' if configs['run'].optimization['validation_reference'] == 'unweighted' else ''
                    min_loss_achieved = diagnostics[val_ref_set_prefix + 'wt_val_loss']['min_tertiary_loss_achieved_all']
                    for step, loss in configs['run'].optimization['validation_milestone'].iteritems():
                        if global_step >= step and min_loss_achieved > loss:
                            raise MilestoneError('Milestone at step ' + str(global_step) + \
                                                 ' missed because minimum loss achieved so far is ' + str(min_loss_achieved))

                    # restart if gradients are zero
                    if (diagnostics['min_grad'] == 0 and diagnostics['max_grad'] == 0) or \
                       (configs['run'].evaluation['include_diagnostics'] and (np.isnan(diagnostics['min_grad']) or np.isnan(diagnostics['max_grad']))):
                        raise DeadGradientError('Gradient is dead.')

                # Predict structures. Currently assumes that weighted training and validation models are available, and fails if they're not.
                if global_step % configs['run'].io['prediction_frequency'] == 0:
                    predict_and_log(log_dir, configs, models, session)

                # Checkpoint
                if global_step % configs['run'].io['checkpoint_frequency'] == 0:
                    models['training'].save(session)

        except tf.errors.OutOfRangeError:
            print('Epoch limit reached.')

        except (tf.errors.InvalidArgumentError, DeadGradientError): # InvalidArgumentError is usually triggered by a nan
            models['training'].finish(session, save=False)

            if args.restart_on_dead_gradient:
                print('Nan or dead gradient encountered; model will be resumed from last checkpoint if one exists, or restarted from scratch otherwise.')        
                if not os.path.isdir(checkpoints_dir):
                    for sub_dir in next(os.walk(run_dir))[1]: rmtree(os.path.join(run_dir, sub_dir)) # erase all old directories    
                restart = True
            else:
                print('Nan or dead gradient encountered; model will be terminated.')        

        except MilestoneError:
            models['training'].finish(session, save=False)

            if args.restart_on_missed_milestone:
                print('Milestone missed; model will be restarted from scratch with an incremented seed.')
                
                for sub_dir in next(os.walk(run_dir))[1]: rmtree(os.path.join(run_dir, sub_dir)) # erase all old directories

                # modify configuration file with new seed
                old_seed = configs['training'].initialization['graph_seed']
                new_seed = old_seed + args.seed_increment
                for line in fileinput.input(args.config_file, inplace=True):
                    print line.replace('randSeed ' + str(old_seed), 'randSeed ' + str(new_seed)),
                
                restart = True
            else:
                print('Milestone missed; model will be terminated.')
            
        except:
            print('Unexpected error: ', sys.exc_info()[0])
            raise

        finally:
            # Wrap up (ask threads to stop, save final checkpoint, etc.)
            if models['training']._is_started: models['training'].finish(session, save=args.checkpoint_on_finish)
            stdout_err_file_handle.close()
    
    return restart

# main
if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser(description="Run RGN model with specified parameters and configuration file.")
    parser.add_argument('-d', '--base_directory',              default='.',         help='parent directory containing runs, data, checkpoints, and logs')
    parser.add_argument('-p', '--prediction_only',             action='store_true', help='if set only a single batch of prediction is made with no training')
    parser.add_argument('-e', '--evaluation_model',            action='append',     help='evaluation model to include (more than one is allowed). ' + \
                                                                                         'must be of the form [un]weighted_[training,validation,testing].')
    parser.add_argument('-m', '--milestone',                   type=lambda m: map(float, m.split(':')), \
                                                               action='append',     help='milestone that the model must achieve or it will be restarted. ' + \
                                                                                         'milestones must be of the form step:loss. multiple milestones can be set.')
    parser.add_argument('-r', '--restart_on_dead_gradient',    action='store_true', help='if a zero gradient or nan (requires include_diagnostics) are encountered, ' + \
                                                                                         'restart from last checkpoint or from scratch if no checkpoint is found. ' + \
                                                                                         'default behavior is for model to terminate.')
    parser.add_argument('-R', '--restart_on_missed_milestone', action='store_true', help='if a validation milestone is missed, restart from scratch with a new seed ' + \
                                                                                         '(incremented by seed_increment). default behavior is for model to terminate.')
    parser.add_argument('-c', '--checkpoint_on_finish',        action='store_true', help='checkpoint when the last epoch is completed.')
    parser.add_argument('-s', '--seed_increment',              type=int, default=8, help='amount to increment seed by if milestones are not met')
    parser.add_argument('-g', '--gpu',                         type=int,            help='GPU device to use')
    gpugrp = parser.add_mutually_exclusive_group()
    gpugrp.add_argument('-a', '--fill_gpu',                    action='store_true', help='fill all available GPU memory')
    gpugrp.add_argument('-f', '--gpu_fraction',                type=float,          help='fill only specified GPU memory fraction')
    parser.add_argument('config_file',                                              help='configuration file containing specification of RGN model')
    args = parser.parse_args()

    # set up signal for premature interruption
    signal.signal(signal.SIGINT, lambda _, __: exit(0))

    # invoke inner loop
    while loop(args): pass
