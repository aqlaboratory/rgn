""" Configuration classes for geometric network models and runs. """

__author__ = "Mohammed AlQuraishi"
__copyright__ = "Copyright 2018, Harvard Medical School"
__license__ = "MIT"

from ast import literal_eval

# helper functions
flt_or_none = lambda x: float(x) if x is not None else None
int_or_none = lambda x: int(x) if x is not None else None
str_or_none = lambda x: None if isinstance(x, basestring) and x == 'none' else x
str_or_bool = lambda x: (x == 'true' or x == 'True') if isinstance(x, basestring) else x
eval_if_str = lambda x: literal_eval(x) if isinstance(x, basestring) else x

def dict_import(file):
    """ Imports configuration dictionary from disk """
    
    vars_ = {}
    with open(file) as f:
        for line in f:
            if line[0] != '#':
                name, var = line.partition(' ')[::2]
                vars_[name.strip()] = var.strip()
    return vars_

class Config(object):
    """Abstract class for encapsulating configuration settings."""

    def __init__(self, file=None, config={}):
        """Loads configuration from disk and calls concrete method to assign values to local attributes"""
        
        # load configuration values from file if available
        if file is not None:
            file_config = dict_import(file)
            file_config.update(config)
            config = file_config

        # assign config values
        self._create_config(config)

    def _create_config(self, config):
        raise NotImplementedError('Abstract method')

class RGNConfig(Config):
    """Encapsulates configuration parameters for recurrent geometric network models

       Options marked with HO indicate that they're completely dependent on higher-order layers being enabled.
       Options marked with pHO indicate that their behavior is partially dependent on higher-order layers.

    """

    def _create_config(self, config):
        # io
        self.io = {'name':                              config.get('name',                  None),
                   'num_edge_residues':             int(config.get('numEdgeResidues',       2)),
                   'num_evo_entries':               int(config.get('numEvoEntries',         20)),
                   'data_files':                        config.get('dataFiles',             None), # a python list of file names, used by default
                   'data_files_glob':                   config.get('dataFilesGlob',         None), # a glob, used if no data_files are supplied
                   'evaluation_sub_groups': eval_if_str(config.get('evaluationSubGroups',   [])),
                   'alphabet_file':                     config.get('alphabetFile',          None), # if passed this overrides alphabet_init
                   'checkpoints_directory':             config.get('checkpointsDirectory',  None),
                   'logs_directory':                    config.get('logsDirectory',         None),
                   'log_model_summaries':   str_or_bool(config.get('logModelSummaries',     True)),
                   'log_alphabet':          str_or_bool(config.get('logAlphabet',           False)),
                   'detailed_logs':         str_or_bool(config.get('detailedLogs',          True)),
                   'max_checkpoints':       int_or_none(config.get('maxCheckpoints',        None)),
                   'checkpoint_every_n_hours':      int(config.get('checkpointEveryNHours', 24))} # this is in addition to the max_checkpoints

        # compute-related issues 
        self.computing = {'num_cpus':                             int(config.get('numCPUs',                        4)),
                          'num_recurrent_shards':                 int(config.get('numRecurrentShards',             1)),
                          'num_recurrent_parallel_iters':         int(config.get('numRecurrentParallelIters',      32)),
                          'default_device':                           config.get('defaultDevice',                  ''),
                          'functions_on_devices':         eval_if_str(config.get('functionsOnDevices',             {'/cpu:0': ['point_to_coordinate']})),
                          'gpu_fraction':                       float(config.get('gpuFraction',                    1)),
                          'allow_gpu_growth':             str_or_bool(config.get('allowGPUGrowth',                 False)),
                          'fill_gpu':                     str_or_bool(config.get('fillGPU',                        False)),
                          'num_reconstruction_fragments': int_or_none(config.get('numReconstructionFragments',     6)),
                          'num_reconstruction_parallel_iters':    int(config.get('numReconstructionParallelIters', 4))}

        # initialization
        self.initialization = {'graph_seed':                        int_or_none(config.get('randSeed',                      None)),
                               'angle_shift':                       eval_if_str(config.get('angleShift',                    [0., 0., 0.])),
                               'recurrent_forget_bias':                   float(config.get('recurrentForgetBias',           1)),                               
                               'recurrent_init':                    eval_if_str(config.get('recurrentInit',                 None)), # can be list if HO
                               'recurrent_seed':                    int_or_none(config.get('recurrentSeed',                 None)),
                               'recurrent_out_proj_init':           eval_if_str(config.get('recurrentOutProjInit',          {'base': {}, 'bias': {}})),
                               'recurrent_out_proj_seed':           int_or_none(config.get('recurrentOutProjSeed',          None)),
                               'recurrent_nonlinear_out_proj_init': eval_if_str(config.get('recurrentNonlinearOutProjInit', {'base': {}, 'bias': {}})),
                               'recurrent_nonlinear_out_proj_seed': int_or_none(config.get('recurrentNonlinearOutProjSeed', None)),
                               'alphabet_init':                     eval_if_str(config.get('alphabetInit',                  {})),
                               'alphabet_seed':                     int_or_none(config.get('alphabetSeed',                  None)),
                               'queue_seed':                        int_or_none(config.get('queueSeed',                     None)),
                               'dropout_seed':                      int_or_none(config.get('dropoutSeed',                   None)),
                               'zoneout_seed':                      int_or_none(config.get('zoneoutSeed',                   None)),
                               'evolutionary_multiplier':                 float(config.get('evolutionaryMultiplier',        1))}

        # optimization
        self.optimization = {'optimizer':                       config.get('optimiser',            'steepest'),
                             'learning_rate':             float(config.get('learnRate',            0.001)), # all optimizers
                             'momentum':                  float(config.get('momentum',             0)),     # momentum, rmsprop, has no analog in autograd
                             'beta1':                     float(config.get('beta1',                0.9)),   # adam, momentum in autograd
                             'beta2':                     float(config.get('beta2',                0.999)), # adam, hoMomentum in autograd
                             'epsilon':                   float(config.get('epsilon',              10e-8)), # adam, rmsprop, adadelta. this should really be 1e-8
                             'decay':                     float(config.get('decay',                0.9)),   # rmsprop, adadelta (rho), momentum in autograd
                             'initial_accumulator_value': float(config.get('initAccumulatorValue', 0.1)),   # adagrad
                             'rescale_behavior':    str_or_none(config.get('rescaleBehavior',      None)),
                             'gradient_threshold':        float(config.get('gradientThreshold',    'inf')),
                             'recurrent_threshold': flt_or_none(config.get('recurrentThreshold',   None)),  # only TF-based RNNs
                             'alphabet_temperature':      float(config.get('alphabetTemperature',  1.0)),
                             'batch_size':                  int(config.get('batchSize',            256)),
                             'num_steps':                   int(config.get('maxSeqLength',         500)),   # Longer seqs removed, shorter ones padded. Max irrespective of curriculum
                             'num_epochs':          int_or_none(config.get('numEpochs',            None))}

        # queueing
        self.queueing = {'file_queue_capacity':        int(config.get('fileQueueCapacity',        1000)),  # Defaults make sense if each file has ~100 sequences
                         'batch_queue_capacity':       int(config.get('batchQueueCapacity',       10000)),
                         'min_after_dequeue':          int(config.get('minAfterDequeue',          500)),
                         'shuffle':            str_or_bool(config.get('shuffle',                  True)),
                         'bucket_boundaries':  eval_if_str(config.get('bucketBoundaries',         None)),
                         'num_evaluation_invocations': int(config.get('numEvaluationInvocations', 1))}

        # curriculum
        self.curriculum = {'mode':                str_or_none(config.get('currMode',            None)),
                           'behavior':            str_or_none(config.get('currBehavior',        None)),
                           'slope':                     float(config.get('currSlope',           1.0)),
                           'base':                      float(config.get('currBase',            4.0)),
                           'rate':                      float(config.get('currRate',            0.002)),
                           'threshold':                 float(config.get('currThreshold',       5.0)),
                           'change_num_iterations':       int(config.get('currChangeNumIters',  5)),
                           'sharpness':                 float(config.get('currSharpness',       20.)),
                           'update_loss_history': str_or_bool(config.get('updateLossHistory',   False)),
                           'loss_history_subgroup':           config.get('lossHistorySubgroup', 'all')}

        # architecture
        self.architecture = {'recurrent_unit':                                       config.get('recurrentUnit',                        'LSTM'),
                             'recurrent_layer_size':                     eval_if_str(config.get('recurrentSize',                        [20])),
                             'recurrent_peepholes':                      str_or_bool(config.get('recurrentPeepholes',                   True)), # LSTM
                             'all_to_all_peepholes':                     str_or_bool(config.get('allToAllPeepholes',                    False)), # LSTM
                             'bidirectional':                            str_or_bool(config.get('bidirectional',                        False)), # pHO
                             'higher_order_layers':                      str_or_bool(config.get('higherOrderLayers',                    False)),
                             'include_recurrent_outputs_between_layers': str_or_bool(config.get('includeRecurrentOutputsBetweenLayers', True)), # HO
                             'include_dihedrals_between_layers':         str_or_bool(config.get('includeDihedralsBetweenLayers',        False)), # HO
                             'residual_connections_every_n_layers':      int_or_none(config.get('residualConnectionsEveryNLayers',      None)), # HO
                             'first_residual_connection_from_nth_layer': int_or_none(config.get('firstResidualConnectionFromNthLayer',  1)), # HO
                             'recurrent_to_output_skip_connections':     str_or_bool(config.get('recurrentToOutputSkipConnections',     False)), # HO
                             'input_to_recurrent_skip_connections':      str_or_bool(config.get('inputToRecurrentSkipConnections',      False)), # HO
                             'all_to_recurrent_skip_connections':        str_or_bool(config.get('allToRecurrentSkipConnections',        False)), # HO
                             'recurrent_nonlinear_out_proj_size':        eval_if_str(config.get('recurrentNonlinearOutputProjSize',     None)),
                             'recurrent_nonlinear_out_proj_function':                config.get('recurrentNonlinearOutputProjFunction', 'tanh'),
                             'tertiary_output':                                      config.get('tertiaryOutput',                       'linear'),
                             'alphabet_size':                            eval_if_str(config.get('alphabetSize',                         None)), # pHO
                             'alphabet_trainable':                       str_or_bool(config.get('alphabetTrainable',                    True)),
                             'include_primary':                          str_or_bool(config.get('includePrimary',                       True)),
                             'include_evolutionary':                     str_or_bool(config.get('includeEvolutionary',                  False))}

        # regularization
        self.regularization = {'recurrent_input_keep_probability':           eval_if_str(config.get('recurInKeepProb',                    1.0)),
                               'recurrent_output_keep_probability':          eval_if_str(config.get('recurOutKeepProb',                   1.0)),
                               'recurrent_keep_probability':                 eval_if_str(config.get('recurKeepProb',                      1.0)),
                               'recurrent_state_zonein_probability':         eval_if_str(config.get('recurStateZoneinProb',               1.0)),
                               'recurrent_memory_zonein_probability':        eval_if_str(config.get('recurMemoryZoneinProb',              1.0)),
                               'alphabet_keep_probability':                  eval_if_str(config.get('alphabetKeepProb',                   1.0)), # pHO
                               'alphabet_normalization':                     str_or_none(config.get('alphabetNormalization',              None)), # pHO
                               'recurrent_nonlinear_out_proj_normalization': str_or_none(config.get('recurNonlinearOutProjNormalization', None)),
                               'recurrent_layer_normalization':              str_or_bool(config.get('recurLayerNormalization',            False)), # LNLSTM
                               'recurrent_variational_dropout':              str_or_bool(config.get('recurVariationalDropout',            False))}

        # loss
        self.loss = {'include':                       str_or_bool(config.get('includeLoss',                 True)),
                     'tertiary_weight':                     float(config.get('tertiaryWeight',              1.0)),
                     'tertiary_normalization':                    config.get('tertiaryNormalization',       'zeroth'),
                     'batch_dependent_normalization': str_or_bool(config.get('batchDependentNormalization', True)),
                     'atoms':                                     config.get('lossAtoms',                   'c_alpha')}

class RunConfig(Config):
    """Encapsulates configuration parameters for an entire run comprised of possibly multiple models"""

    def _create_config(self, config):
        # names
        self.names = {'run':                      config.get('runName'),
                      'dataset':                  config.get('datasetName'),
                      'alphabet':                 config.get('alphabetName', None)}

        # io
        self.io = {'full_training_glob':       config.get('fullTrainingGlob',     '*'),
                   'sample_training_glob':     config.get('sampleTrainingGlob',   '*'),
                   'full_validation_glob':     config.get('fullValidationGlob',   '*'),
                   'sample_validation_glob':   config.get('sampleValidationGlob', '*'),
                   'full_testing_glob':        config.get('fullTestingGlob',      '*'),
                   'sample_testing_glob':      config.get('sampleTestingGlob',    '*'),
                   'evaluation_frequency': int(config.get('evaluationFrequency',  10)),
                   'prediction_frequency': int(config.get('predictionFrequency',  100)),
                   'checkpoint_frequency': int(config.get('checkpointFrequency',  10000))}

        # compute-related issues
        self.computing = {'training_device':   config.get('trainingDevice',   'GPU'),
                          'evaluation_device': config.get('evaluationDevice', 'GPU')}

        # optimization
        self.optimization = {'validation_milestone': eval_if_str(config.get('validationMilestone', {})),
                             'validation_reference':             config.get('validationReference', 'weighted')} # '[un]weighted', used for milestones, curricula, and predictions

        # queueing
        self.queueing = {'training_file_queue_capacity':    int(config.get('trainingFileQueueCapacity',    1000)),
                         'evaluation_file_queue_capacity':  int(config.get('evaluationFileQueueCapacity',  10)),
                         'training_batch_queue_capacity':   int(config.get('trainingBatchQueueCapacity',   10000)),
                         'evaluation_batch_queue_capacity': int(config.get('evaluationBatchQueueCapacity', 300)),
                         'training_min_after_dequeue':      int(config.get('trainingMinAfterDequeue',      500)),
                         'evaluation_min_after_dequeue':    int(config.get('evaluationMinAfterDequeue',    10)),
                         'training_shuffle':        str_or_bool(config.get('trainingShuffle',              True)),
                         'evaluation_shuffle':      str_or_bool(config.get('evaluationShuffle',            False))}

        # evaluation
        self.evaluation = {'num_training_samples':                  int(config.get('numTrainingSamples',                    98)),
                           'num_validation_samples':                int(config.get('numValidationSamples',                  100)),
                           'num_testing_samples':                   int(config.get('numTestingSamples',                     100)),
                           'num_training_invocations':              int(config.get('numTrainingInvocations',                1)),     # evaluation (! actual training)
                           'num_validation_invocations':            int(config.get('numValidationInvocations',              1)), 
                           'num_testing_invocations':               int(config.get('numTestingInvocations',                 1)), 
                           'include_weighted_training':     str_or_bool(config.get('includeWeightedTraining',               False)),  
                           'include_weighted_validation':   str_or_bool(config.get('includeWeightedValidation',             False)),  
                           'include_weighted_testing':      str_or_bool(config.get('includeWeightedTesting',                False)),  
                           'include_unweighted_training':   str_or_bool(config.get('includeUnweightedTraining',             False)),  
                           'include_unweighted_validation': str_or_bool(config.get('includeUnweightedValidation',           False)), 
                           'include_unweighted_testing':    str_or_bool(config.get('includeUnweightedTesting',              False)),
                           'include_diagnostics':           str_or_bool(config.get('includeDiagnostics',                    True))}

        # loss
        self.loss = {'training_tertiary_normalization':          config.get('trainingTertiaryNormalization',         'first'),
                     'evaluation_tertiary_normalization':        config.get('evaluationTertiaryNormalization',       'first'),
                     'training_batch_dependent_normalization':   config.get('trainingBatchDependentNormalization',   True),
                     'evaluation_batch_dependent_normalization': config.get('evaluationBatchDependentNormalization', True)}
