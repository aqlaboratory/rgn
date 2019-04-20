# Recurrent Geometric Networks
This is the reference (TensorFlow) implementation of recurrent geometric networks (RGNs), described in the paper [End-to-end differentiable learning of protein structure](https://www.biorxiv.org/content/early/2018/08/29/265231). 

## Installation and requirements
Extract all files in the [model](https://github.com/aqlaboratory/rgn/tree/master/model) directory in a single location and use `protling.py`, described further below, to train new models and predict structures. Below are the language requirements and package dependencies:

* Python 2.7
* TensorFlow >= 1.4 (tested up to 1.12)
* setproctitle

## Usage
The [`protling.py`](https://github.com/aqlaboratory/rgn/blob/master/model/protling.py) script facilities training of and prediction using RGN models. Below are typical use cases. The script also accepts a number of command-line options whose functionality can be queried using the `--help` option.

#### Train a new model or continue training an existing model
RGN models are described using a configuration file that controls hyperparameters and architectural choices. For a list of available options and their descriptions, see its [documentation](https://github.com/aqlaboratory/rgn/blob/master/CONFIG.md). Once a configuration file has been created, along with a suitable dataset (download a ready-made [ProteinNet](https://github.com/aqlaboratory/proteinnet) data set or create a new one from scratch using the [`convert_to_tfrecord.py`](https://github.com/aqlaboratory/rgn/blob/master/model/convert_to_tfrecord.py) script), the following directory structure must be created:

```
<baseDirectory>/runs/<runName>/<datasetName>/<configurationFile>
<baseDirectory>/data/<datasetName>/[training,validation,testing]
```

Where the first path points to the configuration file and the second path to the directories containing the training, validation, and possibly test sets. Note that `<runName>` and `<datasetName>` are user-defined variables specified in the configuration file that encode the name of the model and dataset, respectively.

Training of a new model can then be invoked by calling:

```
python protling.py [configurationFilePath] -d [baseDirectory]
```

Download a pre-trained model for an example of a correctly defined directory structure. Note that ProteinNet training sets come in multiple "thinnings" and only one should be used at a time by placing it in the main training directory.

To resume training an existing model, run the command above for a previously trained model with saved checkpoints.

#### Predict new structures using a trained model
To predict the structure of a new protein using an existing model with a saved checkpoint, call:

```
python protling.py [configFilePath] -d [baseDirectory] -p
```

This predicts the structures of the dataset specified in the configuration file. By default only the validation set is predicted, but this can be changed using the `-e` option.

## Pre-trained models
Below we make available pre-trained RGN models using the [ProteinNet](https://github.com/aqlaboratory/proteinnet) 7 - 12 datasets as checkpointed TF graphs. These models are identical to the ones used in reporting results in the [bioRxiv preprint](https://www.biorxiv.org/content/early/2018/08/29/265231), except for the CASP 11 model which is slightly different due to using a newer codebase.

| [CASP7](https://sharehost.hms.harvard.edu/sysbio/alquraishi/rgn_models/RGN7.tar.gz) | [CASP8](https://sharehost.hms.harvard.edu/sysbio/alquraishi/rgn_models/RGN8.tar.gz) | [CASP9](https://sharehost.hms.harvard.edu/sysbio/alquraishi/rgn_models/RGN9.tar.gz) | [CASP10](https://sharehost.hms.harvard.edu/sysbio/alquraishi/rgn_models/RGN10.tar.gz) | [CASP11](https://sharehost.hms.harvard.edu/sysbio/alquraishi/rgn_models/RGN11.tar.gz) | [CASP12](https://sharehost.hms.harvard.edu/sysbio/alquraishi/rgn_models/RGN12.tar.gz) |
| --- | --- | --- | --- | --- | --- |

To train new models from scratch using the same hyperparameter choices as the above models, use the appropriate configuration file from [here](https://github.com/aqlaboratory/rgn/tree/master/configurations).

## PyTorch implementation
The reference RGN implementation is currently only available in TensorFlow, however the [OpenProtein](https://github.com/OpenProtein/openprotein) project has implementations of various aspects of the RGN model in PyTorch.

## Reference
[End-to-end differentiable learning of protein structure, Cell Systems 2019](https://www.cell.com/cell-systems/fulltext/S2405-4712(19)30076-6)
