# Recurrent Geometric Networks
This is the reference (TensorFlow) implementation of recurrent geometric networks (RGNs), described in the paper [End-to-end differentiable learning of protein structure](https://www.cell.com/cell-systems/fulltext/S2405-4712(19)30076-6). 

## Installation and requirements
Extract all files in the [model](https://github.com/aqlaboratory/rgn/tree/master/model) directory in a single location and use `protling.py`, described further below, to train new models and predict structures. Below are the language requirements and package dependencies:

* Python 2.7
* TensorFlow >= 1.4 (tested up to 1.12)
* setproctitle

## Usage
The [`protling.py`](https://github.com/aqlaboratory/rgn/blob/master/model/protling.py) script facilities training of and prediction using RGN models. Below are typical use cases. The script also accepts a number of command-line options whose functionality can be queried using the `--help` option.

### Train a new model or continue training an existing model
RGN models are described using a configuration file that controls hyperparameters and architectural choices. For a list of available options and their descriptions, see its [documentation](https://github.com/aqlaboratory/rgn/blob/master/CONFIG.md). Once a configuration file has been created, along with a suitable dataset (download a ready-made [ProteinNet](https://github.com/aqlaboratory/proteinnet) data set or create a new one from scratch using the [`convert_to_tfrecord.py`](https://github.com/aqlaboratory/rgn/blob/master/model/convert_to_tfrecord.py) script), the following directory structure must be created:

```
<baseDirectory>/runs/<runName>/<datasetName>/<configurationFile>
<baseDirectory>/data/<datasetName>/[training,validation,testing]
```

Where the first path points to the configuration file and the second path to the directories containing the training, validation, and possibly test sets. Note that `<runName>` and `<datasetName>` are user-defined variables specified in the configuration file that encode the name of the model and dataset, respectively.

Training of a new model can then be invoked by calling:

```
python protling.py <configurationFilePath> -d <baseDirectory>
```

Download a pre-trained model for an example of a correctly defined directory structure. Note that ProteinNet training sets come in multiple "thinnings" and only one should be used at a time by placing it in the main training directory.

To resume training an existing model, run the command above for a previously trained model with saved checkpoints.

### Predict sequences in ProteinNet TFRecords format using a trained model
To predict the structures of proteins already in ProteinNet `TFRecord` format using an existing model with a saved checkpoint, call:

```
python protling.py <configFilePath> -d <baseDirectory> -p
```

This predicts the structures of the dataset specified in the configuration file. By default only the validation set is predicted, but this can be changed using the `-e` option, e.g. `-e weighted_testing` to predict the test set.

### Predict structure of a single new sequence using a trained model
If all you have is a single sequence for which you wish to make a prediction, there are multiple steps that must be performed. First, a PSSM needs to be created by running JackHMMer (or a similar tool) against a sequence database, the resulting PSSM must be combined with the sequence in a ProteinNet record, and the file must be converted to the `TFRecord` format. Predictions can then be made as previously described.

Below is an example of how to do this using the supplied scripts (in [data_processing](https://github.com/aqlaboratory/rgn/upload/master/data_processing)) and one of the pre-trained models, assumed to be unzipped in `<baseDirectory>`. HMMER must also be installed. The raw sequence databases (`<fastaDatabase>`) used in building PSSMs can be obtained from [here](https://github.com/aqlaboratory/proteinnet/blob/master/docs/raw_data.md). The script below assumes that `<sequenceFile>` only contains a single sequence in the FASTA file format.

```
jackhmmer.sh <sequenceFile> <fastaDatabase>
python convert_to_proteinnet.py <sequenceFile>
python convert_to_tfrecord.py <sequenceFile>.proteinnet <sequenceFile>.tfrecord 42
cp <sequenceFile>.tfrecord <baseDirectory>/data/<datasetName>/testing/
python protling.py <baseDirectory>/runs/<runName>/<datasetName>/<configurationFile> -d <baseDirectory> -p -e weighted_testing
```

The first line searches the supplied database for matches to the supplied sequence and extracts a PSSM out of the results. It will generate multiple new files. These are then used in the second line to construct a text-based ProteinNet file (with 42 entries per evolutionary profile, compatible with the pre-trained RGN models). The third line converts the file to `TFRecords` format, and the fourth line copies the file to the testing directory of a pre-trained model. Finally the fifth line predicts the structure using the pre-trained RGN model. The outputs will be placed in  `<baseDirectory>/runs/<runName>/<datasetName>/<latestIterationNumber>/outputsTesting/` and will be comprised of two files: a `.tertiary` file which contains the atomic coordinates, and `.recurrent_states` file which contains the RGN latent representation of the sequence.

## Pre-trained models
Below we make available pre-trained RGN models using the [ProteinNet](https://github.com/aqlaboratory/proteinnet) 7 - 12 datasets as checkpointed TF graphs. These models are identical to the ones used in reporting results in the [_Cell Systems_ paper](https://www.cell.com/cell-systems/fulltext/S2405-4712(19)30076-6), except for the CASP 11 model which is slightly different due to using a newer codebase.

| [CASP7](https://sharehost.hms.harvard.edu/sysbio/alquraishi/rgn_models/RGN7.tar.gz) | [CASP8](https://sharehost.hms.harvard.edu/sysbio/alquraishi/rgn_models/RGN8.tar.gz) | [CASP9](https://sharehost.hms.harvard.edu/sysbio/alquraishi/rgn_models/RGN9.tar.gz) | [CASP10](https://sharehost.hms.harvard.edu/sysbio/alquraishi/rgn_models/RGN10.tar.gz) | [CASP11](https://sharehost.hms.harvard.edu/sysbio/alquraishi/rgn_models/RGN11.tar.gz) | [CASP12](https://sharehost.hms.harvard.edu/sysbio/alquraishi/rgn_models/RGN12.tar.gz) |
| --- | --- | --- | --- | --- | --- |

To train new models from scratch using the same hyperparameter choices as the above models, use the appropriate configuration file from [here](https://github.com/aqlaboratory/rgn/tree/master/configurations).

## PyTorch implementation
The reference RGN implementation is currently only available in TensorFlow, however the [OpenProtein](https://github.com/OpenProtein/openprotein) project implements various aspects of the RGN model in PyTorch, and [PyTorch-RGN](https://github.com/conradry/pytorch-rgn) is a work-in-progress implementation of the RGN model.

## Reference
[End-to-end differentiable learning of protein structure, Cell Systems 2019](https://www.cell.com/cell-systems/fulltext/S2405-4712(19)30076-6)
