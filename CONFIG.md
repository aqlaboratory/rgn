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
