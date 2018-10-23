""" Neural Network TensorFlow operations for protein structure prediction.

    In general this module contains functions for constructing different parts
    of GeomNetModel networks, excepting ones related to geometric operations.

    There are some conventions used throughout this module. First, most functions
    accept some combination of TF tensors and regular python objects. Since all
    the functions construct parts of TF graphs, the TF tensors they accept are meant
    to be variables that can change from data point to data point or iteration to
    iteration. On the other hand, the python objects are meant to be fixed parameters
    used once in the construction of the TF graph and never revisted. Which is which
    is indicated in each function, and by the fact that python objects are not converted
    into TF tensors. Having said that, some funcs are actually somewhat loose, and
    would work with dynamic values for the supposedly fixed arguments. However the
    intended behavior is what's described.
"""

__author__ = "Mohammed AlQuraishi"
__copyright__ = "Copyright 2018, Harvard Medical School"
__license__ = "MIT"

# Imports
import numpy as np
import tensorflow as tf

# Constants
NUM_AAS = 20
NUM_DIMENSIONS = 3
NUM_DIHEDRALS = 3

### Public functions
# These functions expose a public interface that properly encapsulates their internals
# using tensorflow scoping operations and such. While they are primarily used by the
# GeomNetModel, they may also have general utility beyond it. All these functions
# are strictly stateless, possessing no internal TF variables.

def masking_matrix(mask, name=None):
    """ Constructs a masking matrix to zero out pairwise distances due to missing residues or padding. 

        This function needs to be called for each individual sequence, and so it's folded in the reading/queuing
        pipeline for performance reasons.

    Args:
        mask: 0/1 vector indicating whether a position should be masked (0) or not (1)

    Returns:
        A square matrix with all 1s except for rows and cols whose corresponding indices in mask are set to 0.
        [MAX_SEQ_LENGTH, MAX_SEQ_LENGTH]

    """

    with tf.name_scope(name, 'masking_matrix', [mask]) as scope:
        mask = tf.convert_to_tensor(mask, name='mask')

        mask = tf.expand_dims(mask, 0)
        base = tf.ones([tf.size(mask), tf.size(mask)])
        matrix_mask = base * mask * tf.transpose(mask)

        return matrix_mask

def effective_steps(masks, num_edge_residues, name=None):
    """ Returns the effective number of steps, i.e. number of residues that are non-missing and are not just
        padding, given a masking matrix.

    Args:
        masks: A batch of square masking matrices (batch is last dimension)
        [MAX_SEQ_LENGTH, MAX_SEQ_LENGTH, BATCH_SIZE]

    Returns:
        A vector with the effective number of steps
        [BATCH_SIZE]

    """

    with tf.name_scope(name, 'effective_steps', [masks]) as scope:
        masks = tf.convert_to_tensor(masks, name='masks')
        
        traces = tf.matrix_diag_part(tf.transpose(masks, [2, 0, 1]))
        eff_stepss = tf.add(tf.reduce_sum(traces, [1]), num_edge_residues, name=scope) # NUM_EDGE_RESIDUES shouldn't be here, but I'm keeping it for 
                                                                                       # legacy reasons. Just be clear that it's _always_ wrong to have
                                                                                       # it here, even when it's not equal to 0.

        return eff_stepss

def read_protein(filename_queue, max_length, num_edge_residues, num_evo_entries, name=None):
    """ Reads and parses a protein TF Record. 

        Primary sequences are mapped onto 20-dimensional one-hot vectors.
        Evolutionary sequences are mapped onto num_evo_entries-dimensional real-valued vectors.
        Secondary structures are mapped onto ints indicating one of 8 class labels.
        Tertiary coordinates are flattened so that there are 3 times as many coordinates as 
        residues.

        Evolutionary, secondary, and tertiary entries are optional.

    Args:
        filename_queue: TF queue for reading files
        max_length:     Maximum length of sequence (number of residues) [MAX_LENGTH]. Not a 
                        TF tensor and is thus a fixed value.

    Returns:
        id: string identifier of record
        one_hot_primary: AA sequence as one-hot vectors
        evolutionary: PSSM sequence as vectors
        secondary: DSSP sequence as int class labels
        tertiary: 3D coordinates of structure
        matrix_mask: Masking matrix to zero out pairwise distances in the masked regions
        pri_length: Length of amino acid sequence
        keep: True if primary length is less than or equal to max_length

    """

    with tf.name_scope(name, 'read_protein', []) as scope:
        # Set up reader and read
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        # Parse TF Record
        context, features = tf.parse_single_sequence_example(serialized_example,
                                context_features={'id': tf.FixedLenFeature((1,), tf.string)},
                                sequence_features={
                                    'primary':      tf.FixedLenSequenceFeature((1,),               tf.int64),
                                    'evolutionary': tf.FixedLenSequenceFeature((num_evo_entries,), tf.float32, allow_missing=True),
                                    'secondary':    tf.FixedLenSequenceFeature((1,),               tf.int64,   allow_missing=True),
                                    'tertiary':     tf.FixedLenSequenceFeature((NUM_DIMENSIONS,),  tf.float32, allow_missing=True),
                                    'mask':         tf.FixedLenSequenceFeature((1,),               tf.float32, allow_missing=True)})
        id_ = context['id'][0]
        primary =   tf.to_int32(features['primary'][:, 0])
        evolutionary =          features['evolutionary']
        secondary = tf.to_int32(features['secondary'][:, 0])
        tertiary =              features['tertiary']
        mask =                  features['mask'][:, 0]

        # Predicate for when to retain protein
        pri_length = tf.size(primary)
        keep = pri_length <= max_length

        # Convert primary to one-hot
        one_hot_primary = tf.one_hot(primary, NUM_AAS)

        # Generate tertiary masking matrix. If mask is missing then assume all residues are present
        mask = tf.cond(tf.not_equal(tf.size(mask), 0), lambda: mask, lambda: tf.ones([pri_length - num_edge_residues]))
        ter_mask = masking_matrix(mask, name='ter_mask')        

        # Return tuple
        return id_, one_hot_primary, evolutionary, secondary, tertiary, ter_mask, pri_length, keep

def curriculum_weights(base, slope, max_seq_length, name=None):
    ''' Returns a tensor of weights that correspond to the current curriculum, as parametrized by base and slope.

    Args:
        base: Value of the base parameter, a TF tensor that is expected to change as training progresses.
        slope: Value of the slope parameter. Not a TF tensor and is thus a fixed value.
        max_seq_length: Maximum length of sequences. Not a TF tensor and is thus a fixed value.

    Returns:
        [MAX_SEQ_LENGTH - 1]

    '''

    with tf.name_scope(name, 'curriculum_weights', [base]) as scope:
        base = tf.convert_to_tensor(base, name='base')

        steps = tf.to_float(tf.range(max_seq_length - 1)) # The minus one factor is because we ignore self-distances.
        weights = tf.sigmoid(-(slope * (steps - base)), name=scope) 

        return weights

def weighting_matrix(weights, name=None):
    """ Takes a vector of weights and returns a weighting matrix in which the ith weight is 
        in the ith upper diagonal of the matrix. All other entries are 0.

        This functions needs to be called once per curriculum update / iteration, but then used for 
        the entire batch.

        This function intimately mixes python and TF code. It can do so because all the python code
        needs to be run only once during the initial construction phase and does not rely on any
        tensor values. This interaction is subtle however.

    Args:
        weights: Curriculum weights. A TF tensor that is expected to change as curriculum progresses. [MAX_SEQ_LENGTH - 1]

    Returns
        [MAX_SEQ_LENGTH, MAX_SEQ_LENGTH]

    """

    with tf.name_scope(name, 'weighting_matrix', [weights]) as scope:
        weights = tf.convert_to_tensor(weights, name='weights')

        max_seq_length = weights.get_shape().as_list()[0] + 1
        split_indices = np.diag_indices(max_seq_length)   

        flat_indices = []
        flat_weights = []
        for i in range(max_seq_length - 1):
            indices_subset = np.concatenate((split_indices[0][:-(i+1), np.newaxis], split_indices[1][i+1:, np.newaxis]), 1)
            weights_subset = tf.fill([len(indices_subset)], weights[i])
            flat_indices.append(indices_subset)
            flat_weights.append(weights_subset)
        flat_indices = np.concatenate(flat_indices)
        flat_weights = tf.concat(flat_weights, 0)

        mat = tf.sparse_to_dense(flat_indices, [max_seq_length, max_seq_length], flat_weights, validate_indices=False, name=scope)

        return mat

def id_filter(ids, filter_string, delimiter='#', name=None):
    """ Returns a boolean mask corresponding to the chosen id filter from a list of ids """

    with tf.name_scope(name, 'id_filter', [ids, filter_string]) as scope:
        ids           = tf.convert_to_tensor(ids,           name='ids')
        filter_string = tf.convert_to_tensor(filter_string, name='filter_string')

        return tf.equal(tf.string_split(ids, delimiter=delimiter).values[0::2], filter_string, name=scope)
