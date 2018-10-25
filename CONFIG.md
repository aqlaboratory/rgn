# Configuration Files
Configuration files specify hyperparameter and architectural choices of an RGN model. They are comprised of a list of option specifications in the following format:

```
# Optional comment
camelCaseOption <optionValue>
```

Below are the major options along with their descriptions and allowed values. Not all options are documented. For a complete list see [config.py](https://github.com/aqlaboratory/rgn/blob/master/model/config.py).

## IO
| Option Name | Acceptable Values | Description |
| --- | --- | --- |
| runName | string | user-specified model name |
| datasetName | string | user-specified dataset name |
| numEvoEntries | integer | number of entries present in evolutionary profiles |
| maxSeqLength | integer | longest acceptable protein (longer proteins will be ignored) |
| trainingShuffle | boolean | if True shuffle training set |
| evaluationShuffle | boolean | if True shuffle evaluation set |
| evaluationFrequency | integer | number of iterations between evaluations | 
| predictionFrequency | integer | number of iterations between predicting structures | 
| checkpointFrequency | integer | number of iterations between model checkpoints | 
| numTrainingSamples | integer | number of samples when evaluating training set | 
| numValidationSamples | integer | number of samples when evaluating validation set | 
| numTestingSamples | integer | number of samples when evaluating test set | 
| numTrainingInvocations | integer | number of batches to process when evaluating training set | 
| numValidationInvocations | integer | number of batches to process when evaluating validation set | 
| numTestingInvocations | integer | number of batches to process when evaluating test set | 

## Architecture
| Option Name | Acceptable Values | Description |
| --- | --- | --- |
| includePrimary | boolean | if True include primary sequence as input |
| includeEvolutionary | boolean | if True include PSSM as input |
| tertiaryOutput | linear, linear_alphabet, angular, angular_alphabet | form of output units in last layer |
| alphabetSize | integer | alphabet size if using alphabetized output |
| alphabetTrainable | boolean | if True alphabet is trainable |
| recurrentNonlinearOutputProjSize | integer | if set specifies size of non-linear projection layer after last RNN layer |
| recurrentNonlinearOutputProjFunction | tanh, relu | type of non-linearity to use |
| recurrentUnit | LSTM, GRU, CudnnLSTM, CudnnGRU | type of RNN unit |
| recurrentSize | [integer, ...] | list of RNN layer sizes |
| bidirectional | boolean | whether RNN is uni- or bi-directional |
| higherOrderLayers | boolean | if True enables construction of more complex RNN architectures (all options below require this), and integrates RNN directions before passing on to next layer |
| includeRecurrentOutputsBetweenLayers | boolean | if True passes raw outputs of one layer to another |
| includeDihedralsBetweenLayers | boolean | if True passes dihedral outputs of one layer to another |
| residualConnectionsEveryNLayers | integer | connect layers residually every Nth layer |
| firstResidualConnectionFromNthLayer | integer | begin residual connections at the specified layer |
| recurrentToOutputSkipConnections | boolean | use skip connections from all hidden layers to final layer |
| inputToRecurrentSkipConnections | boolean | use skip connections from input layer to all hidden layers |
| allToRecurrentSkipConnections | boolean | use skip connections from all layers to final |

## Regularization
All keep probabilities correspond to 1 - dropout probability, and can be specified as a single real number to be used for all layers, or as a list of real numbers, one per layer.

| Option Name | Acceptable Values | Description |
| --- | --- | --- |
| recurInKeepProb | real or list of reals | keep probabilit(y,ies) of inputs to recurrent layers |
| recurOutKeepProb | real or list of reals | keep probabilit(y,ies) of outputs of recurrent layers |
| recurKeepProb | real or list of reals | keep probabilit(y,ies) of states of recurrent layers |
| recurStateZoneinProb | real or list of reals | zone in probabilit(y,ies) of states of recurrent layers |
| recurMemoryZoneinProb | real or list of reals | zone in probabilit(y,ies) of memories of recurrent layers |
| recurStateZoneinProb | real or list of reals | zone in probabilit(y,ies) of states of recurrent layers |
| recurVariationalDropout | boolean | if True uses variational dropout for recurrent state (requires recurKeepProb > 0) |
| alphabetKeepProb | real or list of reals | keep probabilit(y,ies) of alphabet |
| alphabetNormalization | \[batch,layer\]\_normalization | normalization for alphabet layer, if not None |

## Optimization
| Option Name | Acceptable Values | Description |
| --- | --- | --- |
| batch_size | integer | batch size |
| bucketBoundaries | list of integers | specifies buckets (protein lengths) to use during batching | 
| optimiser | steepest, momentum, rmsprop, adam, adagrad, adadelta | optimizer to use |
| learning_rate | real | optimizer learning rate |
| momentum | real | momentum in steepest and momentum optimizers |
| beta1 | real | beta1 in adam optimizer |
| beta2 | real | beta2 in adam optimizer |
| epsilon | real | epsilon in rmsprop, adam, and adadelta optimizers |
| decay | real | decay in rmsprop and adadelta (rho) optimizers |
| initAccumulatorValue | real | initial accumulator value in adagrad optimizer |
| rescaleBehavior | norm_rescaling or hard_clipping | gradient rescaling approach |
| gradientThreshold | real | threshold to use when rescaling gradients |
| recurrentThreshold | real | threshold for clipping RNN cells |
| alphabetTemperature | real between 0 and 1 | temperature of alphabet softmax |
| numEpochs | integer | number of epochs to train for |
| validationMilestone | {iteration:drmsd, ...} | dictionary of (validation) dRMSDs that must be reached by corresponding iterations, otherwise training is restarted with a new seed |

## Initialization
Many initialization options accept an initialization dictionary of the form `{'base': spec, 'bias': spec}`, where `'base'` controls the overall distribution and `'bias'` the bias terms, and `spec` is of the form `{'center': real, 'range': real, 'dist': <dist>}` where ``<dist>`` can be one of `'gaussian'`, `'uniform'`, `'orthogonal'`, `'gaussian_variance_scaling'`, and `'uniform_variance_scaling'`. Additional `spec` terms may be specifiable for some distributions.

| Option Name | Acceptable Values | Description |
| --- | --- | --- |
| randSeed | integer | random seed for initializing model |
| recurrentInit | one or a list of initialization dictionaries | initialization scheme for recurrent layers |
| recurrentOutProjInit | initialization dictionary | initialization scheme for output projection layer |
| recurrentNonlinearOutProjInit | initialization dictionary | initialization scheme for non-linear output projection layer |
| alphabetInit | initialization spec | initialization scheme for alphabet |
| recurrentForgetBias | real | initial value of forget bias in LSTM units |

## Compute
Additional compute-related options can be specified as command-line options to `protling.py`.

| Option Name | Acceptable Values | Description |
| --- | --- | --- |
| trainingDevice | CPU, GPU | where to place training model |
| evaluationDevice | CPU, GPU | where to place evaluation model |
