__author__ = "Mohammed AlQuraishi"
__copyright__ = "Copyright 2018, Harvard Medical School"
__license__ = "MIT"

from copy import deepcopy
from glob import glob
from ast import literal_eval

import numpy as np
import numpy.random as npr
import tensorflow as tf
import time
import os

from model import RGNModel
from config import RGNConfig


# Constants and shared templates used by most / all test functions
base_dir = '../'
train_dir = base_dir + 'data/unofficial/tfrecord/training3/'
train_files = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
eval_dir = base_dir + 'data/unofficial/tfrecord/test3/'
eval_files = ['1', '2', '3']
checkpoints_dir = base_dir + 'checkpoints/'
artifacts_dir = base_dir + 'artifacts/'
alphabets_dir = base_dir + 'data/unofficial/alphabets/'

state_size = 15
train_batch_size = 10
eval_batch_size = 26
max_seq_length = 100
input_size = 20
output_size = 3
num_cpus = 1
rand_seed = 1
use_gpu = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

c_train_template = RGNConfig(config={'checkpointsDirectory':  checkpoints_dir,
                                     'logModelSummaries':     False,
                                     'recurrentSize':         [state_size],
                                     'batchSize':             train_batch_size,
                                     'maxSeqLength':          max_seq_length,
                                     'minAfterDequeue':       20,
                                     'randSeed':              rand_seed,
                                     'numCPUs':               num_cpus,
                                     'shuffle':               False})
c_train_template.io['data_files'] = [os.path.join(train_dir, file) for file in train_files]

c_eval_template = deepcopy(c_train_template)
c_eval_template.io['data_files'] = [os.path.join(eval_dir, file) for file in eval_files]
c_eval_template.optimization['batch_size'] = eval_batch_size
c_eval_template.optimization['min_after_dequeue'] = 1

npr.seed(1)
w_template = {'rnn/lstm_cell/kernel':     (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
              'rnn/lstm_cell/bias':       (npr.rand(state_size * 4) - 0.5) * 0.15,
              'rnn/lstm_cell/w_f_diag':   (npr.rand(state_size) - 0.5) * 0.05,
              'rnn/lstm_cell/w_i_diag':   (npr.rand(state_size) - 0.5) * 0.05,
              'rnn/lstm_cell/w_o_diag':   (npr.rand(state_size) - 0.5) * 0.05,
              'linear_dihedrals/weights': (npr.rand(state_size, output_size) - 0.5) * 0.05,
              'linear_dihedrals/biases':  ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}


# Helper functions
def assign_weights(session, weight_dict, scope='RGN'):
    """ Assigns variables passed weights

        Args:
            weight_dict: dict of variable names (under given scope) and their values
    """

    with tf.variable_scope(scope, reuse=True):
        assignments = []
        for key, val in weight_dict.iteritems():
            var = tf.get_variable(key)
            assignments.append(var.assign(val.astype('float32')))
        session.run(assignments)

def get_node_ops(nodes, scope='RGN'):
    """ Returns handles to ops (values) of nodes by name that can be evaluated by sess.run() 

        Args:
            nodes: list of strings containing node names
    """

    g = tf.get_default_graph()

    return [g.get_operation_by_name(scope + '/' + node).outputs[0] for node in nodes]

def get_var_ops(vars_, scope='RGN'):
    """ Returns handles to ops (values) of variables by name that can be evaluated by sess.run() 

        Args:
            vars_: list of strings containing variable names
    """

    with tf.variable_scope(scope, reuse=True): 
        var_ops = [tf.get_variable(var) for var in vars_]

    return var_ops

def dicts_to_matched_tuples(dict1, dict2):
    """ Converts pair of dicts to pair of matched tuples so that their elements can be compared. 

        Throws an exception if the set of keys in both dicts are not the same.
    """

    try:
        return [(dict1[k], dict2[k]) for k in set(dict1.keys() + dict2.keys())]
    except KeyError:
        raise RuntimeError('Dictionaries are not comparable.')


# Test classes
class CanonicalTest(tf.test.TestCase):
    """ These tests check the fundamental correctness of the forward pass (any desired nodes and variables) 
        and backward pass (by evaluating desired nodes and variables after training). They also test 
        checkpointing support by checkpointing every step of the way and testing that it's the same state
        as if it were not checkpointed. 

        Tests (and their names) are by default for tertiary outputs. Secondary stuff is named explicitly. 

        All tests are done with a learning rate of 0.001 and are only run for a few iterations. This is by
        design, as the accumulation of numerical differences between different hardware platforms very 
        rapidly leads to divergence, and so numbers can only be related directly at the very beginning. 

        Some tests involve random behavior (all the dropout and zoneout ones), which tends to change from 
        one TF release to the other. These tests really can't be used in an absolute manner. Their purpose
        is to test for localized changes before/after some thing is introduced, but in general they always
        be stale and need to be calibrated (i.e. numberic values computed) before and after a test.

        When reference is made to the numbers having been obtained by Autograd, it specifically means the
        Jupyter notebook titled '1.0 (validation against iteration 3, v3)'. """

    def setUp(self):
        super(CanonicalTest, self).setUp()
        tf.logging.error("Starting: %s", self._testMethodName)

    def tearDown(self):
        super(CanonicalTest, self).tearDown()
        tf.logging.error("Finished: %s", self._testMethodName)

    def _createModel(self, c_train, c_evals):
        m_train = RGNModel('training', c_train)
        m_evals = []
        for c_eval in c_evals: m_evals.append(RGNModel('evaluation', c_eval))

        return m_train, m_evals

    def _runModel(self, sess, iteration, m_train, m_evals, weight_dict, node_dict, variable_dict, rtol, atol, restart_every_iteration, scope='RGN'):
        # fetch nodes and expected values
        nodes = node_dict.keys()
        vars_ = variable_dict.keys()
        valuess_expected = node_dict.values() + variable_dict.values()
        num_entries = len(nodes) + len(vars_)

        # set up
        if iteration == 0 or restart_every_iteration: m_train.start(m_evals, sess)
        if iteration == 0: assign_weights(sess, weight_dict)
        if restart_every_iteration:
            for _ in range(iteration): m_train.diagnose(sess) # do vacuous diagnostics to bring state of queue back to where it's supposed to be

        # test losses and nodes
        for config in range(len(m_evals)):
            # get handles to nodes / variables
            node_ops = get_node_ops(nodes, scope + '/model_' + str(config + 1))
            var_ops = get_var_ops(vars_)
            ops = node_ops + var_ops

            # run session to get actual values
            values_actual = sess.run(ops)

            # test agreement
            for entry in range(num_entries): self.assertAllClose(valuess_expected[entry][iteration][config], values_actual[entry], rtol, atol)

            # run evaluation in case things like loss history need to get updated
            m_evals[config].evaluate(sess)

        # train for one step
        m_train.train(sess)

    def _testCore(self, c_train, c_evals, weight_dict=None, node_dict={}, variable_dict={}, use_gpu=False, 
                  rtol=1e-2, atol=1e-2, restart_every_iteration=False, scope='RGN'):
        """ Canonical test that checks any node/variable(s) based on an evaluation model coupled with a training model.

            Runs training for as many iterations given minus 1 (first set is tested before training.)
            All dicts must have the same number of iterations. One of the dicts must be non-empty.
            Checkpoints and restores at every iteration.

            Args:
                c_train:       configuration object for training model
                c_evals:       configuration objects for evaluation model
                weight_dict:   {'variable': replacement_value} (dict of weights to replace model weights)
                node_dict:     {'node': expected_value ([ITERATION, CONFIG, NODE_VALUE]), ...}
                variable_dict: {'variable': expected_value ([ITERATION, CONFIG, VARIABLE_VALUE]), ...}
        """

        # make sure number of iterations between and within dicts match and that some dict was passed
        num_iterationss = set([len(val) for dict_ in [node_dict, variable_dict] for val in dict_.values()])
        if len(num_iterationss) > 1:
            raise RuntimeError('Number of iterations do not match in dicts.')
        elif len(num_iterationss) == 0:
            raise RuntimeError('No non-empty dicts passed.')
        num_iterations = num_iterationss.pop()

        # prepare config
        config = tf.ConfigProto(inter_op_parallelism_threads=num_cpus, intra_op_parallelism_threads=num_cpus)

        # iterate, each time creating a new model (and session!), testing current iteration, checkpointing, and destroying it.
        # if restart_every_iteration is set to False, then a single session/model is reused. Ugliness with two "withs" insures that.
        if restart_every_iteration:
            try:
                for iteration in range(num_iterations):
                    with tf.Graph().as_default() as g:
                        m_train, m_evals = self._createModel(c_train, c_evals)
                        with self.test_session(use_gpu=use_gpu, graph=g, config=config) as sess:
                            try:
                                self._runModel(sess, iteration, m_train, m_evals, weight_dict, node_dict, variable_dict, rtol, atol, restart_every_iteration, scope)
                            finally:
                                m_train.finish(sess, save=True, close_session=False, reset_graph=False)
            
            finally:
                for f in glob(checkpoints_dir + '*'): os.remove(f) # remove all checkpoints

        else:
            with tf.Graph().as_default() as g:
                m_train, m_evals = self._createModel(c_train, c_evals)
                with self.test_session(use_gpu=use_gpu, graph=g, config=config) as sess:
                    try:
                        for iteration in range(num_iterations):
                            self._runModel(sess, iteration, m_train, m_evals, weight_dict, node_dict, variable_dict, rtol, atol, restart_every_iteration, scope)
                    finally:
                        m_train.finish(sess, save=False, close_session=False, reset_graph=False)

    def testBasic(self):
        # values sourced from autograd
        self._testCore(c_train_template, [c_eval_template], w_template,
                       {'all/loss': [[1117.3124301441253], [1528.6822316674461], [8918.7761559069913]],
                        'drmsds': [[[1259.07786793,   803.25406635,  1205.25539413,  1025.61087985, 1676.55300594,   1308.98564775,   1153.30808244,   854.51602809,  815.66860265,  1004.5901707 ,   1196.04068161,   1451.05497598,  1286.43255861,   1176.03539134,    589.94585875,   1106.00902386,  1658.48477058,   1238.3419573 ,   1041.211955  ,    908.69045666,  696.5834041 ,    844.03238061,   1333.24227589,    919.05794194, 1259.16899425,   1238.97081143]],
                                   [[1852.03230968,  1298.21547954,  1701.79515487,  1546.85502048, 2432.2781724 ,   1926.46521634,   1672.05336474,  1102.57260516,  990.31405429,  1403.04726402,   1860.18546927,   2213.50226333,  1885.86486933,   1644.2164523 ,    639.60193536,   1603.07148149,  2406.45469603,   1746.76373983,   1470.15775544,   1242.50577499,  811.55817662,   1237.96052076,   1937.86330937,   1344.84908538,  849.91827047,    925.63558189]],
                                   [[10398.96744671, 9410.69248146,  9624.58117421,  9229.32326745, 12797.198093 ,  11281.55467097,  10034.56338289,   6177.7895715, 5779.85300392,  7942.10719805,  11387.75744491,  12228.81485384, 10716.13548079,   9030.89986508,   4245.08641234,   9650.40891751, 12667.68625944,   9579.74124615,   8347.40016602,   7528.6852053 , 5495.06240634,   8121.44738695,  10675.87702396,   8034.47102478, 8552.71371788,   2949.36235213]]]})

    def testConstantLossCurriculum(self):
        c_train = deepcopy(c_train_template)
        c_train.curriculum['mode'] = 'loss'
        c_train.curriculum['behavior'] = 'constant'
        c_train.curriculum['slope'] = 1.0
        c_train.curriculum['base'] = 40.0

        c_eval_wt_val = deepcopy(c_train)
        c_eval_wt_val.io['data_files'] = c_eval_template.io['data_files']
        c_eval_wt_val.optimization['batch_size'] = c_eval_template.optimization['batch_size']

        c_eval_unwt_val = deepcopy(c_eval_wt_val)
        c_eval_unwt_val.curriculum['mode'] = None
        c_eval_unwt_val.curriculum['behavior'] = None

        # values sourced from autograd
        self._testCore(c_train, [c_eval_wt_val, c_eval_unwt_val], w_template,
                       {'all/loss': [[811.62446809010203, 1117.3124300232862], [2856.1590681258494, 4500.3302033531345], [3716.1991858584784, 5724.4626765986459]],
                        'drmsds': [[[  712.4029018,   656.46970474,   790.84479972,   652.82797692,   961.66260809,   767.57335486,   874.35258872,   806.63533543,   782.99623018,   749.18344096,   738.81951414,   774.31806268,   827.08632311,   867.14142333,   589.94743697,   820.71294084,   942.09196495,   918.12720509,   813.31124771,   822.39872585,   698.84897879,   722.90436389,   770.49497554,   649.19058281, 1152.92809206,  1238.96539117],
                                    [1259.07786823,   803.25406637,  1205.25539413,  1025.61087986,   1676.55300594,  1308.98564716,  1153.30808244,  854.51602809,   815.66860265,  1004.5901706 ,  1196.04068161,  1451.05497598,   1286.43255741,  1176.03539134,   589.9458586 ,  1106.00902386, 1658.48476905,  1238.3419567 ,  1041.21195485,   908.69045584,   696.5834041 ,   844.03238061,  1333.24227729,   919.05794194, 1259.16899449,  1238.97081143]],
                                   [[3072.38719134,  3008.42793815,  2911.85857299,  3065.61335124,   3168.97061669,  2941.77352692,  3017.70879495,  2891.57130501, 2701.93006486,  3019.22871584,  3068.32486473,  3267.78362357,   3045.9070289 ,  3093.56330986,  2006.90778597,  3057.33263582, 3186.76070856,  3166.60588075,  3106.64842597,  3006.92982167,  2557.87728753,  2975.23974785,  3033.74464829,  2996.44252548, 2141.96776781,   748.62963052],
                                    [5391.52600272,  4645.21763053,  4942.86150264,  4718.65664836,   6729.95540672,  5780.03519699,  5098.49233851,  3129.42496819, 2864.29983114,  4069.45413954,  5812.61878454,  6396.09591494,   5526.21617409,  4662.83236075,  2006.95721157,  4901.05258194, 6663.49016856,  4959.97668451,  4286.68446429,  3785.27389317,  2631.89622374,  4053.00748602,  5547.72873732,  4090.80613275, 3565.38418254,   748.64062109]],
                                   [[3980.29953437,  3912.8153788 ,  3805.71332237,  3963.61695076,   4084.56028093,  3856.92137197,  3913.6971335 ,  3685.5340745 , 3475.78707043,  3886.0444654 ,  3987.26781491,  4192.61316339,   3952.33294312,  3974.66567725,  2623.42857364,  3952.23612967, 4103.54137995,  4054.2940867 ,  3977.94455372,  3861.36371322,  3325.75878519,  3850.13789212,  3941.71257486,  3871.03295692, 3033.58241754,  1354.2765871],
                                    [6779.16037791,  5967.68230448,  6239.49896096,  5970.19469129,   8414.64044261,  7304.24708855,  6465.52745619,  3971.47672996, 3668.21995854,  5143.51939118,  7359.1793521 ,  8014.77664466,   6965.92066737,  5872.49712966,  2623.4918399 ,  6216.04403989, 8331.61846207,  6240.0050447 ,  5410.11199782,  4822.33365404,  3422.75419481,  5180.21693498,  6969.75994666,  5185.21810334, 4943.64177028,  1354.29240765]]]})

    def testFixedRateLossCurriculum(self):
        c_train = deepcopy(c_train_template)
        c_train.curriculum['mode'] = 'loss'
        c_train.curriculum['behavior'] = 'fixed_rate'
        c_train.curriculum['slope'] = 1.0
        c_train.curriculum['base'] = 40.0
        c_train.curriculum['rate'] = 10.0

        c_eval_wt_val = deepcopy(c_train)
        c_eval_wt_val.io['data_files'] = c_eval_template.io['data_files']
        c_eval_wt_val.optimization['batch_size'] = c_eval_template.optimization['batch_size']

        # values sourced from autograd, tests for 5 rather than the usual 2 steps.
        self._testCore(c_train, [c_eval_wt_val], w_template,
                       {'all/loss': [[811.62446811817381], [3513.1184983200842], [6271.9810019918259], [983.22696333058173], [7074.5178111699024], [2538.085871467898]],
                        'drmsds': [[[712.40290179,   656.46970474,   790.84479972,   652.82797677,  961.66260809,   767.57335483,   874.35258872,   806.63533543,  782.99623018,   749.18344109,   738.81951414,   774.31806268,  827.08632313,   867.14142333,   589.94743712,   820.71294084,  942.09196521,   918.12720528,   813.31124777,   822.39872622,  698.84897879,   722.90436389,   770.49497536,   649.19058281,  1152.92809197,  1238.96539117]],
                                   [[3921.80539858,  3732.46069854,  3801.21695969,  3821.12079274, 4076.66666828,  3775.84066084,  3927.1942207 ,  3128.3883932 , 2864.1464899 ,  3721.93243559,  3946.11086323,  4173.37973482, 3970.63112941,  3887.23580532,  2006.95720964,  3846.72458199, 4123.69043465,  3932.20050842,  3826.52653447,  3578.26011741, 2631.8595079 ,  3657.03015323,  3932.74326478,  3679.85911881, 2628.45865349,   748.64062071]],
                                   [[7176.87079058,  6775.09118146,  6970.02912915,  6839.79661145, 7616.0410783 ,  7143.09359067,  7070.5124654 ,  4843.62164305, 4502.20906871,  6236.79189278,  7311.1771536 ,  7627.26801766, 7289.57180087,  6815.54972764,  3263.56443027,  6908.3615865 , 7661.42704304,  7001.39493587,  6502.73596029,  5890.93469423, 4241.59846174,  6276.34647895,  7266.07296404,  6280.0842309 , 5576.21178568,  1985.14932895]],
                                   [[872.05246464,   881.82019819,   1007.73195078,   808.44389826, 1057.3778369 ,  1032.04209123,  1009.80330527,   833.45110999, 896.68374053,   858.34141002,   905.1993699 ,   802.57994523,  968.65739504,   958.83681926,   755.93807196,   969.78985402,  1024.05648003,   971.160418  ,   864.71197301,   888.99793349, 882.77021036,   852.37783662,   947.02304745,   780.45504825,  2086.33507447,  1647.2635637 ]],
                                   [[8398.63187157,  7506.85882369,  7750.86366995,  7426.18411137, 9732.63014403,  8924.43051309,  8061.12738794,  4952.44467959, 4605.14278385,  6387.44268979,  8985.64632406,  9526.27290827, 8618.55531257,  7281.33804213,  3342.34373111,  7753.27837412, 9677.59414824,  7730.25859655,  6725.28662141,  6025.29046531, 4341.75807969,  6495.56539188,  8614.38793767,  6455.58384599, 6551.86494493,  2066.68169162]],
                                   [[3130.43086019,  2486.70577288,  2843.25274224,  2679.18857745, 3930.38493036,  3305.35334123,  2886.62016529,  1785.88342616, 1592.80776533,  2328.36017386,  3285.65021109,  3733.89590438, 3179.72105306,  2701.13694644,  1049.61700489,  2775.29773013, 3896.26613357,  2879.07616255,  2472.42446129,  2108.22341618, 1381.79318992,  2234.402753  ,  3229.59646555,  2317.7659459 , 1411.33222366,   365.04530159]]]},
                       {'curriculum_step': [[40.0], [50.0], [60.0], [70.0], [80.0], [90.0]]})

    def testLossThresholdLossCurriculumAndHistoryUpdating(self):
        c_train = deepcopy(c_train_template)
        c_train.curriculum['mode'] = 'loss'
        c_train.curriculum['behavior'] = 'loss_threshold'
        c_train.curriculum['slope'] = 1.0
        c_train.curriculum['base'] = 40.0
        c_train.curriculum['rate'] = 10.0
        c_train.curriculum['threshold'] = 20.0
        c_train.curriculum['change_num_iterations'] = 3 # don't need this here but want to lower it to test loss history pruning functionality

        c_eval_wt_val = deepcopy(c_train)
        c_eval_wt_val.io['data_files']                  = c_eval_template.io['data_files']
        c_eval_wt_val.optimization['batch_size']        = c_eval_template.optimization['batch_size']
        c_eval_wt_val.curriculum['update_loss_history'] = True

        # values sourced from autograd, tests for 5 rather than the usual 2 steps. Need to lower tolerances because at the last step numerical differences really add up.
        # The semantics of this section are not necessarily how the TF-based scheme will ultimately work, 
        # but I just want something that's equivalent to the autograd version so that I can check for bugs.
        self._testCore(c_train, [c_eval_wt_val], w_template,
                       {'all/loss': [[811.62415], [3513.1182], [5560.0815], [963.36652], [7626.5435], [2066.1833]],
                        'drmsds': [[[712.40222168,   656.46929932,   790.84454346,   652.82727051, 961.66308594,   767.57336426,   874.35150146,   806.63531494, 782.99615479,   749.18304443,   738.8190918 ,   774.31762695, 827.08624268,   867.14123535,   589.94744873,   820.71191406, 942.09136963,   918.12640381,   813.31054688,   822.39825439, 698.84887695,   722.9039917 ,   770.49517822,   649.19024658, 1152.92797852,  1238.96508789]],
                                   [[3921.80639648,  3732.46069336,  3801.21337891,  3821.11621094, 4076.66333008,  3775.84350586,  3927.1940918 ,  3128.38793945, 2864.14770508,  3721.93359375,  3946.11035156,  4173.37646484, 3970.63085938,  3887.23779297,  2006.95812988,  3846.72192383, 4123.68847656,  3932.19848633,  3826.52734375,  3578.26293945, 2631.85766602,  3657.02758789,  3932.74584961,  3679.85839844, 2628.4597168 ,   748.64080811]],
                                   [[6142.89794922,  5929.84082031,  5976.91601562,  5987.69580078, 6362.39697266,  6034.35644531,  6127.70849609,  4842.17578125, 4501.9921875 ,  5765.0546875 ,  6213.86572266,  6460.97753906, 6199.27099609,  6019.67285156,  3263.56030273,  6033.19921875, 6407.52539062,  6099.56591797,  5907.97119141,  5584.98681641, 4241.53613281,  5743.53027344,  6160.00488281,  5744.94140625, 4825.33740234,  1985.14648438]],
                                   [[928.00964355,   795.63299561,   995.62634277,   879.63861084, 1120.62487793,   873.32885742,  1068.05432129,   929.09063721, 864.12731934,  1014.23864746,   910.83300781,  1027.66723633, 1048.03662109,  1076.75512695,   592.13397217,   972.62237549, 1142.36987305,  1084.02697754,  1037.42272949,   967.07122803, 715.41455078,   893.19940186,   990.04785156,   890.34857178, 1103.18408203,  1128.02478027]],
                                   [[8705.54785156,  8271.04882812,  8440.6171875 ,  8291.15527344, 9224.08300781,  8721.00976562,  8589.49316406,  5836.70898438, 5446.38476562,  7508.82617188,  8899.77832031,  9239.61132812, 8833.609375  ,  8240.75195312,  3963.18164062,  8400.33398438, 9261.33007812,  8468.88964844,  7857.50976562,  7128.921875  , 5168.27001953,  7619.57910156,  8807.45410156,  7584.81396484, 7092.76708984,  2688.44848633]],
                                   [[2453.92456055,  2132.84936523,  2364.79101562,  2296.09277344, 2641.95166016,  2309.75073242,  2421.15307617,  1637.79394531, 1429.55749512,  2093.01586914,  2442.55615234,  2702.20092773, 2519.05786133,  2397.20825195,   934.54754639,  2311.28295898, 2643.15991211,  2428.40185547,  2299.63598633,  1921.87255859, 1223.37658691,  2042.30212402,  2525.42553711,  2113.75927734, 1033.14233398,   401.95651245]]]},
                       {'curriculum_step': [[40.0], [50.0], [50.0], [50.0], [60.0], [60.0]],
                        'curriculum_loss_history': [[[-1., -1., -1.]], [[-1., -1., 8.1162415]], [[-1., 8.1162415, 35.131182]], [[8.1162415, 35.131182, 55.600815]], [[35.131182, 55.600815, 9.6336652]], [[55.600815, 9.6336652, 76.265435]]]},
                       rtol=1e-1, atol=1e-1)

    def testLossChangeLossCurriculumAndHistoryUpdating(self):
        c_train = deepcopy(c_train_template)
        c_train.curriculum['mode'] = 'loss'
        c_train.curriculum['behavior'] = 'loss_change'
        c_train.curriculum['slope'] = 1.0
        c_train.curriculum['base'] = 40.0
        c_train.curriculum['rate'] = 10.0
        c_train.curriculum['change_num_iterations'] = 5
        c_train.curriculum['sharpness'] = 0.1

        c_eval_wt_val = deepcopy(c_train)
        c_eval_wt_val.io['data_files']                  = c_eval_template.io['data_files']
        c_eval_wt_val.optimization['batch_size']        = c_eval_template.optimization['batch_size']
        c_eval_wt_val.curriculum['update_loss_history'] = True

        npr.seed(1)
        w = {'rnn/lstm_cell/kernel':    (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'rnn/lstm_cell/bias':     (npr.rand(state_size * 4) - 0.5) * 0.15,
             'rnn/lstm_cell/w_f_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_i_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_o_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'linear_dihedrals/weights': (npr.rand(state_size, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':  ((npr.rand(output_size) - 0.5) * 0.05) + 0.2,
             'curriculum_loss_history':  np.array([-1., -1, 10., 10., 10.])}

        # values sourced from autograd, tests for 5 rather than the usual 2 steps. Need to lower tolerances because at the last step numerical differences really add up.
        # The semantics of this section are not necessarily how the TF-based scheme will ultimately work, 
        # but I just want something that's equivalent to the autograd version so that I can check for bugs.
        self._testCore(c_train, [c_eval_wt_val], w,
                       {'all/loss': [[811.62446811817381], [2856.1590688980946], [4344.9414086697989], [2582.6861198060337], [6981.8546053318314], [1736.7348480158662]],
                        'drmsds': [[[712.40290179,   656.46970474,   790.84479972,   652.82797677,  961.66260809,   767.57335483,   874.35258872,   806.63533543,  782.99623018,   749.18344109,   738.81951414,   774.31806268,  827.08632313,   867.14142333,   589.94743712,   820.71294084,  942.09196521,   918.12720528,   813.31124777,   822.39872622,  698.84897879,   722.90436389,   770.49497536,   649.19058281,  1152.92809197,  1238.96539117]],
                                   [[3072.38719134,  3008.42793991,  2911.85857315,  3065.61335431, 3168.97061669,  2941.77352842,  3017.70879495,  2891.57130501,  2701.93006486,  3019.22871584,  3068.32486473,  3267.78362481,  3045.9070289 ,  3093.56330986,  2006.90778867,  3057.33263582,  3186.7607088 ,  3166.60588134,  3106.6484276 ,  3006.92982362,  2557.87729174,  2975.23974889,  3033.74464829,  2996.44252548,  2141.96776781,   748.62963052]],
                                   [[4781.97681807,  4610.20183203,  4625.65332841,  4681.39396872, 4943.34334964,  4645.67152515,  4773.59273225,  3960.64553748,  3666.32370734,  4547.68215512,  4817.98980224,  5047.5953698 ,  4803.93737397,  4733.75131504,  2623.49181102,  4714.21210284,  4970.23894949,  4784.20792262,  4662.23881565,  4426.60728458,  3422.24637855,  4520.60771824,  4770.66415055,  4526.47017802,  3553.4400979 ,  1354.29240071]],
                                   [[2967.80588702,  2734.33762731,  2902.14601824,  2851.89942391, 3131.98661834,  2820.8206015 ,  2971.57460481,  2179.09722376,  1959.44547603,  2757.53174217,  2975.04859064,  3209.30119375,  3053.09449444,  2927.63784721,  1312.8612172 ,  2865.69773776,  3197.78200022,  2980.01981997,  2870.4279071 ,  2573.26712122,  1742.72926226,  2649.08371411,  3019.5858932 ,  2709.04971701,  1583.53807956,   204.06929621]],
                                   [[8031.97246028,  7560.39327618,  7765.67768081,  7607.87563196, 8553.09061973,  8036.12360195,  7888.60174855,  5300.36816215,  4937.32372891,  6823.85976053,  8202.36916539,  8542.55818638,  8139.08316523,  7552.77427996,  3598.95793179,  7713.18990218,  8574.9111079 ,  7783.99178953,  7155.05125404,  6448.80506888,  4667.93060252,  6927.8142766 ,  8121.28587583,  6894.19364023,  6381.82601701,  2318.19080412]],
                                   [[2138.72860162,  1634.22427187,  2027.21185137,  1895.23934007, 2390.65907149,  2008.27670831,  1991.1240206 ,  1342.20067604,  1186.5824904 ,  1712.56864805,  2053.95511362,  2307.17673441,  2120.20372179,  1949.00693428,   784.49138156,  1932.92812112,  2337.1205167 ,  2055.09641836,  1808.83662235,  1508.9627241 ,   991.81131387,  1598.10387413,  2173.9986364 ,  1685.60747253,   813.74856755,   707.24221584]]]},
                       {'curriculum_step': [[40.0], [40.0], [47.62279905095761], [54.627107638682787], [61.798503765010253], [68.57091859250194]],
                        'curriculum_loss_history': [[[-1., -1., 10., 10., 10.]], [[-1., 10., 10., 10., 8.11624146]], [[10.0, 10.0, 10.0, 8.1162447068527701, 28.561590903221521]], [[10.0, 10.0, 8.1162447068527701, 28.561590903221521, 43.449414498621017]], [[10.0, 8.1162447068527701, 28.561590903221521, 43.449414498621017, 25.826861006832079]], [[8.1162447068527701, 28.561590903221521, 43.449414498621017, 25.826861006832079, 69.818545616970383]]]},
                       rtol=1e-3, atol=1)

    def testVacuousHardClipping(self):
        # just testing that the gradients are unperturbed if the threshold is very high, to insure basic code correctness
        c_train = deepcopy(c_train_template)
        c_train.optimization['rescale_behavior'] = 'hard_clipping'
        c_train.optimization['gradient_threshold'] = 1e20

        # values sourced from testBasic
        self._testCore(c_train, [c_eval_template], w_template,
                       {'all/loss':   [[1117.3124301441253], [1528.6822316674461], [8918.7761559069913]],
                        'drmsds': [[[1259.07786793,   803.25406635,  1205.25539413,  1025.61087985, 1676.55300594,   1308.98564775,   1153.30808244,   854.51602809,  815.66860265,  1004.5901707 ,   1196.04068161,   1451.05497598,  1286.43255861,   1176.03539134,    589.94585875,   1106.00902386,  1658.48477058,   1238.3419573 ,   1041.211955  ,    908.69045666,  696.5834041 ,    844.03238061,   1333.24227589,    919.05794194, 1259.16899425,   1238.97081143]],
                                   [[1852.03230968,  1298.21547954,  1701.79515487,  1546.85502048, 2432.2781724 ,   1926.46521634,   1672.05336474,  1102.57260516,  990.31405429,  1403.04726402,   1860.18546927,   2213.50226333,  1885.86486933,   1644.2164523 ,    639.60193536,   1603.07148149,  2406.45469603,   1746.76373983,   1470.15775544,   1242.50577499,  811.55817662,   1237.96052076,   1937.86330937,   1344.84908538,  849.91827047,    925.63558189]],
                                   [[10398.96744671, 9410.69248146,  9624.58117421,  9229.32326745, 12797.198093 ,  11281.55467097,  10034.56338289,   6177.7895715, 5779.85300392,  7942.10719805,  11387.75744491,  12228.81485384, 10716.13548079,   9030.89986508,   4245.08641234,   9650.40891751, 12667.68625944,   9579.74124615,   8347.40016602,   7528.6852053 , 5495.06240634,   8121.44738695,  10675.87702396,   8034.47102478, 8552.71371788,   2949.36235213]]]})

    def testVacuousNormRescaling(self):
        # just testing that the gradients are unperturbed if the threshold is very high, to insure basic code correctness
        c_train = deepcopy(c_train_template)
        c_train.optimization['rescale_behavior'] = 'norm_rescaling'
        c_train.optimization['gradient_threshold'] = 1e20

        # values sourced from testBasic
        self._testCore(c_train, [c_eval_template], w_template,
                       {'all/loss':   [[1117.3124301441253], [1528.6822316674461], [8918.7761559069913]],
                        'drmsds': [[[1259.07786793,   803.25406635,  1205.25539413,  1025.61087985, 1676.55300594,   1308.98564775,   1153.30808244,   854.51602809,  815.66860265,  1004.5901707 ,   1196.04068161,   1451.05497598,  1286.43255861,   1176.03539134,    589.94585875,   1106.00902386,  1658.48477058,   1238.3419573 ,   1041.211955  ,    908.69045666,  696.5834041 ,    844.03238061,   1333.24227589,    919.05794194, 1259.16899425,   1238.97081143]],
                                   [[1852.03230968,  1298.21547954,  1701.79515487,  1546.85502048, 2432.2781724 ,   1926.46521634,   1672.05336474,  1102.57260516,  990.31405429,  1403.04726402,   1860.18546927,   2213.50226333,  1885.86486933,   1644.2164523 ,    639.60193536,   1603.07148149,  2406.45469603,   1746.76373983,   1470.15775544,   1242.50577499,  811.55817662,   1237.96052076,   1937.86330937,   1344.84908538,  849.91827047,    925.63558189]],
                                   [[10398.96744671, 9410.69248146,  9624.58117421,  9229.32326745, 12797.198093 ,  11281.55467097,  10034.56338289,   6177.7895715, 5779.85300392,  7942.10719805,  11387.75744491,  12228.81485384, 10716.13548079,   9030.89986508,   4245.08641234,   9650.40891751, 12667.68625944,   9579.74124615,   8347.40016602,   7528.6852053 , 5495.06240634,   8121.44738695,  10675.87702396,   8034.47102478, 8552.71371788,   2949.36235213]]]})

    def testHardClipping(self):
        c_train = deepcopy(c_train_template)
        c_train.optimization['rescale_behavior'] = 'hard_clipping'
        c_train.optimization['gradient_threshold'] = 5.

        # values sourced from autograd
        self._testCore(c_train, [c_eval_template], w_template,
                       {'all/loss':   [[1117.3124301441253], [1082.019141581744], [1046.3059378055416]],
                        'drmsds': [[[1259.07786793,   803.25406635,  1205.25539413,  1025.61087985, 1676.55300594,   1308.98564775,   1153.30808244,   854.51602809,  815.66860265,  1004.5901707 ,   1196.04068161,   1451.05497598,  1286.43255861,   1176.03539134,    589.94585875,   1106.00902386,  1658.48477058,   1238.3419573 ,   1041.211955  ,    908.69045666,  696.5834041 ,    844.03238061,   1333.24227589,    919.05794194, 1259.16899425,   1238.97081143]],
                                   [[1200.58643465,   767.14882956,  1160.94532394,   976.97575234, 1598.79633006,  1252.09849709,  1108.93633918,   836.69741614, 808.01893081,   969.90533415,  1132.58622694,  1370.29243295, 1229.41248625,  1133.39622557,   595.70727708,  1063.55096168, 1580.95346649,  1190.73561468,  1002.62179294,   884.26879175, 698.82269104,   814.77155534,  1273.78566927,   881.56595458, 1324.7012497 ,  1275.21609692]],
                                   [[1138.08837351,   734.86908456,  1115.60345924,   926.70244641, 1514.53427032,  1193.24059371,  1064.41611266,   820.09752721, 803.07754372,   934.99638486,  1066.20632486,  1281.35126343, 1169.62944539,  1089.2058818 ,   604.89654274,  1020.84739114, 1496.73123472,  1140.78048782,   962.86315705,   861.53186261, 705.3587654 ,   787.53845894,  1210.40657093,   843.72522403, 1401.35365466,  1315.90232125]]]})

    def testNormRescaling(self):
        c_train = deepcopy(c_train_template)
        c_train.optimization['rescale_behavior'] = 'norm_rescaling'
        c_train.optimization['gradient_threshold'] = 5.

        # values sourced from autograd
        self._testCore(c_train, [c_eval_template], w_template,
                       {'all/loss':   [[1117.3124301441253], [1103.7467995276024], [1090.6268919760546]],
                        'drmsds': [[[1259.07786793,   803.25406635,  1205.25539413,  1025.61087985, 1676.55300594,   1308.98564775,   1153.30808244,   854.51602809,  815.66860265,  1004.5901707 ,   1196.04068161,   1451.05497598,  1286.43255861,   1176.03539134,    589.94585875,   1106.00902386,  1658.48477058,   1238.3419573 ,   1041.211955  ,    908.69045666,  696.5834041 ,    844.03238061,   1333.24227589,    919.05794194, 1259.16899425,   1238.97081143]],
                                   [[1236.62254446,   789.10404169,  1188.28134281,  1007.10462488, 1647.30102998,  1287.19631473,  1136.19942893,   847.48791921, 812.44785497,   991.2802773 ,  1171.80601661,  1420.52385302, 1264.83986904,  1159.85756059,   591.81894445,  1089.59874033, 1629.43751769,  1220.21613727,  1026.33140215,   899.1403652 , 697.01224433,   832.47068821,  1310.67857263,   904.6924793 , 1283.39426095,  1252.57275698]],
                                   [[1214.55338547,   775.85189654,  1171.7961667 ,   989.05498372, 1618.40089741,  1265.96856849,  1119.68582585,   840.83787645, 809.63525946,   978.39032372,  1148.1171705 ,  1390.26496931, 1243.71877623,  1144.10498505,   594.02304684,  1073.76393243, 1600.7354524 ,  1202.51135757,  1011.86937797,   890.11087714, 697.91322935,   821.54020517,  1288.51390435,   890.77872221, 1307.96839532,  1266.18960572]]]})

    def testAdam(self):
        # even though this goes for 5 steps numerically it's pretty close to autograd, likely because the learning rate is effectively lowered due to adam.
        c_train = deepcopy(c_train_template)
        c_train.optimization['optimizer'] = 'adam'
        c_train.optimization['beta1'] = 0.8
        c_train.optimization['beta2'] = 0.9
        c_train.optimization['epsilon'] = 1e-2

        # values sourced from autograd
        self._testCore(c_train, [c_eval_template], w_template,
                       {'all/loss':   [[1117.3124301441253], [1109.5024872681829], [1102.5982411194682], [1095.3293954637738], [1087.8150428067247], [1080.4758826574334]],
                        'drmsds': [[[1259.07786793,   803.25406635,  1205.25539413,  1025.61087985, 1676.55300594,   1308.98564775,   1153.30808244,   854.51602809,  815.66860265,  1004.5901707 ,   1196.04068161,   1451.05497598,  1286.43255861,   1176.03539134,    589.94585875,   1106.00902386,  1658.48477058,   1238.3419573 ,   1041.211955  ,    908.69045666,  696.5834041 ,    844.03238061,   1333.24227589,    919.05794194, 1259.16899425,   1238.97081143]],
                                   [[1246.35181303,   794.99840231,  1195.53547274,  1014.93704385, 1659.66888162,  1296.49301358,  1143.47956138,   850.52092244, 813.81452009,   997.00292871,  1182.15985379,  1433.59989048, 1273.93218266,  1166.64804637,   590.96981027,  1096.56590286, 1641.6797659 ,  1227.89543484,  1032.72019757,   903.22743985, 696.77562921,   837.41367339,  1320.30091648,   910.75779054, 1272.935075  ,  1246.68050001]],
                                   [[1235.00369154,   787.82087771,  1186.90225009,  1005.47455997, 1644.60277582,  1285.40060253,  1134.80107332,   847.00490814, 812.24018321,   990.23101668,  1169.80330007,  1418.01857311, 1262.84135764,  1158.33994906,   591.9917647 ,  1088.24666242, 1626.65228435,  1218.63299891,  1025.21108792,   898.39923067, 697.08028685,   831.62565721,  1308.75632227,   903.43581678, 1285.4284869 ,  1253.6085512]],
                                   [[1222.93479822,   780.38346179,  1177.7905119 ,   995.50467325, 1628.56758306,  1273.67694093,  1125.66130135,   843.35427724, 810.67721628,   983.10042859,  1156.76388628,  1401.36816303, 1251.10816086,  1149.57440536,   593.18175904,  1079.48892143, 1610.65858641,  1208.80569168,  1017.2707488 ,   893.36512835, 697.54655163,   825.59405612,  1296.50545607,   895.7464456 , 1298.92898365,  1261.00614513]],
                                   [[1210.33279124,   772.84425908,  1168.34748063,   985.15997453, 1611.78443553,  1261.50383843,  1116.21956789,   839.6320141 , 809.16626064,   975.72432077,  1143.20923144,  1383.89548319, 1238.90227883,  1140.48045461,   594.54313984,  1070.44500532, 1593.91664136,  1198.58202423,  1009.03741721,   888.22810473, 698.19389698,   819.44276008,  1283.72544431,   887.80268758, 1313.27667096,  1268.79492949]],
                                   [[1197.90396387,   765.64384048,  1159.0967857 ,   975.00626659, 1595.16209374,  1249.5629781 ,  1107.00971646,   836.0502321 , 807.79802406,   968.5111957 ,  1129.88892189,  1366.57504156, 1226.89169173,  1131.55925277,   596.01053982,  1061.6251761 , 1577.32834734,  1188.53402497,  1000.97778752,   883.28302256, 698.99978749,   813.53220436,  1271.12195107,   880.03973838, 1327.70300067,  1276.55736403]]]})

    def testRMSProp(self):
        # even though this goes for 5 steps numerically it's pretty close to autograd, likely because the learning rate is effectively lowered due to adam.
        c_train = deepcopy(c_train_template)
        c_train.optimization['optimizer'] = 'rmsprop'
        c_train.optimization['momentum'] = 0.
        c_train.optimization['decay'] = 0.9
        c_train.optimization['epsilon'] = 1e-7

        # values sourced from autograd
        self._testCore(c_train, [c_eval_template], w_template,
                       {'all/loss':   [[1117.3124301441253], [1094.6801080511977], [1071.9299886869696], [1057.5378133745546], [1042.595174571801], [1034.146123743918]],
                        'drmsds': [[[1259.07786793,   803.25406635,  1205.25539413,  1025.61087985, 1676.55300594,   1308.98564775,   1153.30808244,   854.51602809,  815.66860265,  1004.5901707 ,   1196.04068161,   1451.05497598,  1286.43255861,   1176.03539134,    589.94585875,   1106.00902386,  1658.48477058,   1238.3419573 ,   1041.211955  ,    908.69045666,  696.5834041 ,    844.03238061,   1333.24227589,    919.05794194, 1259.16899425,   1238.97081143]],
                                   [[1221.88592844,   779.72237283,  1176.91620894,   994.53622734, 1627.21294742,  1272.65097515,  1124.83105345,   842.9730673 , 810.50354036,   982.3820933 ,  1155.56103698,  1399.89715765, 1250.06855272,  1148.76054707,   593.30025627,  1078.75744935, 1609.27496855,  1207.95964276,  1016.53041318,   892.8684937 , 697.59872127,   825.05663595,  1295.40601181,   895.02456757, 1300.19371769,  1261.8102223]],
                                   [[1183.29901357,   757.49406065,  1148.24879408,   962.97326484, 1575.6718929 ,  1235.64780899,  1096.35502301,   831.89660262, 806.32349853,   960.12568979,  1114.08778596,  1345.94245992, 1212.80247079,  1121.02321482,   597.89544619,  1051.42316705, 1557.78758121,  1176.77830516,   991.48248898,   877.64853319, 700.16450967,   806.7969043 ,  1256.24425745,   870.92319896, 1345.07176002,  1286.07197322]],
                                   [[1158.09241614,   744.39703642,  1130.02164363,   942.75173474, 1541.81239567,  1211.91630849,  1078.34272934,   825.20614817, 804.31315962,   946.10438692,  1087.35254535,  1310.21325204, 1188.7004625 ,  1103.26203192,   601.52735046,  1034.15670426, 1524.02511251,  1156.66830832,   975.40487007,   868.47956357, 702.72160154,   795.79670173,  1230.73570433,   855.67127   , 1375.86534254,  1302.44436746]],
                                   [[1131.29732878,   731.8155468 ,  1110.95440879,   921.46766672, 1505.31554555,  1186.94798243,  1059.68995661,   818.53479476, 802.81288612,   931.46978651,  1059.19239559,  1271.5568542 , 1163.20460648,  1084.53464289,   606.04701648,  1016.37397111, 1487.65890562,  1135.47219956,   958.67018528,   859.36080901, 706.31584412,   784.93306968,  1203.56723872,   839.89450584, 1410.04517819,  1320.341213]],
                                   [[1115.79123978,   725.28320285,  1100.03007171,   909.26708888, 1483.87031279,  1172.7565705 ,  1049.2546344 ,   814.94283743, 802.27005835,   923.16042503,  1043.0071014 ,  1249.04122525, 1148.45841044,  1073.82076271,   609.00866203,  1006.38253399, 1466.10998644,  1123.26457438,   949.17534614,   854.41781476, 708.87327373,   779.14551311,  1187.78125063,   830.92027024, 1430.78013917,  1330.98591119]]]})

    def testFirstOrderLoss(self):
        c_train = deepcopy(c_train_template)
        c_train.loss['tertiary_normalization'] = 'first'

        c_eval = deepcopy(c_eval_template)
        c_eval.loss['tertiary_normalization'] = 'first'

        # values sourced from autograd
        self._testCore(c_train, [c_eval], w_template,
                       {'all/loss':   [[1165.3967160378857], [2968.2763504342415], [9421.5383972698928]],
                        'drmsds': [[[1259.07786793,     803.25406635,  1205.25539413,   1025.61087985, 1676.55300594,   1308.98564775,   1153.30808244,    854.51602809,  815.66860265,  1004.5901707 ,   1196.04068161,   1451.05497598,  1286.43255861,   1176.03539134,    589.94585875,   1106.00902386,  1658.48477058,   1238.3419573 ,   1041.211955  ,    908.69045666,  696.5834041 ,    844.03238061,   1333.24227589,    919.05794194, 1259.16899425,   1238.97081143]],
                                   [[3387.02044353,    2743.1780901 ,  3089.20093899,   2921.76018629, 4306.85999926,   3589.43909117,   3140.64429883,   1940.97512697, 1741.85512523,  2539.97297659,   3580.28376334,   4050.60268628,  3461.24484545,   2936.84435105,   1155.41072488,   3015.65063738,  4263.24140312,   3126.77185414,   2674.57197922,   2311.68776031, 1529.04647413,   2437.90760133,   3502.44147005,   2525.40677861, 1644.56968933,   276.85563487]],
                                   [[10335.42438478,   9350.08793556,  9565.29551022,   9171.9816219 , 12720.21689772,  11211.69520987,  9971.87182409,   6139.82267082, 5743.39234124,  7893.86868953,  11316.47901757,  12154.06895751, 10650.16784814,   8975.18094312,   4217.64946045,   9590.05964154, 12591.6224519 ,   9521.09786095,   8295.76592294,   7481.99119901, 5459.62405934,   8070.14911508,  10610.58588098,   7984.68759192, 8489.44331681,   2922.15944393]]]})

    def testSecondOrderLoss(self):
        c_train = deepcopy(c_train_template)
        c_train.loss['tertiary_normalization'] = 'second'

        c_eval = deepcopy(c_eval_template)
        c_eval.loss['tertiary_normalization'] = 'second'

        # values sourced from autograd
        self._testCore(c_train, [c_eval], w_template,
                       {'all/loss':   [[1208.8778719851207], [4172.8165549812493], [8736.7085158190694]],
                        'drmsds': [[[1259.07786793,    803.25406635,  1205.25539413,   1025.61087985, 1676.55300594,   1308.98564775,   1153.30808244,    854.51602809,  815.66860265,    1004.5901707 ,   1196.04068161,   1451.05497598,  1286.43255861,   1176.03539134,    589.94585875,   1106.00902386,  1658.48477058,   1238.3419573 ,   1041.211955  ,    908.69045666,  696.5834041 ,    844.03238061,   1333.24227589,    919.05794194, 1259.16899425,   1238.97081143]],
                                   [[4472.99792146,   3777.11663423,  4093.91607881,   3898.8281757 , 5624.82322386,   4777.95780015,   4202.48820727,    2580.53267239, 2344.11795152,   3368.47725564,   4793.21372112,   5326.43640425,  4583.43588795,   3873.27141689,   1609.59689231,   4037.31231375,  5568.23017714,   4120.20644709,   3546.39279697,   3107.86344206, 2121.35378927,   3312.43209574,   4612.61901063,   3375.12486934, 2669.69063778,   372.4621525]],
                                   [[9169.4939181 ,   8247.60698834,  8485.95240836,   8128.68257352, 11320.5002812 ,   9933.50527783,   8828.8357481 ,   5449.60120363, 5084.44416419,   7020.66073135,  10012.92563969,  10786.47304456,  9451.54828964,   7962.01849428,   3725.8640704 ,   8488.62530126, 11208.18734898,   8454.48458042,   7350.70944317,   6635.25639126, 4819.91079607,   7134.67192398,   9420.20478026,   7083.44397957, 7337.84790856,   2434.36576206]]]})

    def testBidirectionality(self):
        c_train = deepcopy(c_train_template)
        c_train.architecture['bidirectional'] = True

        c_eval = deepcopy(c_eval_template)
        c_eval.architecture['bidirectional'] = True

        npr.seed(1)
        w = {'bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,
           
             'bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,
           
             'linear_dihedrals/weights':                (npr.rand(state_size * 2, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':                 ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

        # values sourced from autograd
        self._testCore(c_train, [c_eval], w,
                       {'all/loss':   [[1159.2424227882905], [2941.3173292415704], [8497.5744167319554]],
                        'drmsds': [[[1325.4560415 ,   849.4061764 ,  1256.87899964,  1083.22630441,  1765.11321778,   1375.50856652,  1207.42840876,   876.523428  , 827.33137002,  1045.45431626,  1269.26929926,  1540.74182854, 1352.94650529,  1226.88381722,   586.41402574,  1156.58790394, 1743.71041878,  1292.44037028,  1085.66037882,   938.47206502, 698.11272718,   881.52474369,  1401.6154606 ,   964.10056848, 1189.96319062,  1199.53285977]],
                                   [[3595.76131038,  2941.4114417 ,  3281.24339055,  3109.99454642,  4561.19469831,   3818.39571137,  3343.73052938,  2061.88762597, 1854.79848746,  2698.20886298,  3813.90057564,  4299.74442699, 3676.36736806,  3115.19623964,  1239.69771369,  3212.16084552, 4516.80528902,  3319.00625824,  2842.40105425,  2462.61966725, 1638.00186673,  2608.00584062,  3714.75799564,  2688.63756762, 1833.59725679,   226.72399007]],
                                   [[9918.58933468,  8952.59069162,  9176.94079926,  8796.51937957, 12218.92350567,  10755.31931812,   9559.9827963 ,   5893.13110397, 5506.3973903 ,   7582.1031883 ,  10850.59082285,  11665.04905976, 10217.72346162,   8608.70925565,   4039.71197761,   9195.48430789, 12097.81036358,   9139.1117249 ,   7955.98830914,   7179.4971963 , 5227.11424296,   7736.59951981,  10184.379717  ,   7659.47643575, 8075.85901917,   2743.33191322]]]})

    def testAngularOutput(self):
        c_train = deepcopy(c_train_template)
        c_train.architecture['tertiary_output'] = 'angular'

        c_eval = deepcopy(c_eval_template)
        c_eval.architecture['tertiary_output'] = 'angular'

        # values sourced from autograd
        self._testCore(c_train, [c_eval], w_template,
                       {'all/loss':   [[4118.0773529063754], [2105.4722571152138], [9002.3804406430554]],
                        'drmsds': [[[4961.45637158,    4229.27855508,  4537.01747348,   4324.82893137,  6199.88879912,  5302.95801263,  4670.23618762,  2869.70278039,  2615.78217087,  3733.72284027,  5328.65820994,  5888.76989019,  5073.13404098,  4283.07068134,  1817.11412043,  4489.90287628,  6140.30914577,  4560.05282478,  3938.49927889,  3462.36427887,  2385.96214861,  3701.77145762,  5103.04948372,  3747.74993642,  3137.95420086,   566.77647844]],
                                   [[2693.13413527,    1949.09266028,  2365.70328868,   2193.22886811,  3280.45417608,  2759.80634861,  2386.19873671,  1513.48563021,  1347.90873093,  1929.0620499 ,  2681.7294226 ,  3131.24461453,  2586.17301654,  2206.23167372,   892.82187084,  2300.50631751,  3198.95154843,  2388.24195103,  2107.58097225,  1721.31577333,  1142.93723301,  1866.06124545,  2686.32967843,  1876.86389548,   992.90585404,   544.30899303]],
                                   [[10535.82818385,   9535.48557972,  9726.19670881,   9333.82934846,  12990.63562458, 11441.25673136, 10157.01034859, 6160.82949748,  5743.83474227,  7983.58210612,  11560.91456058, 12409.48959805, 10870.76943748, 9122.3677051 ,  4108.13250607,  9775.43112893,  12868.04801776, 9690.58468484,  8415.61765158,  7564.14502483,  5452.3263204 ,  8187.17922162,  10823.47207625, 8087.16888994,  8691.66766426,   2826.08809779]]]})

    def testTrainableLinearAlphabetizedOutput(self):
        alphabet_size = 7

        c_train = deepcopy(c_train_template)
        c_train.architecture['tertiary_output'] = 'linear_alphabet'
        c_train.architecture['alphabet_size'] = alphabet_size

        c_eval = deepcopy(c_eval_template)
        c_eval.architecture['tertiary_output'] = 'linear_alphabet'
        c_eval.architecture['alphabet_size'] = alphabet_size

        npr.seed(1)
        w = {'rnn/lstm_cell/kernel':    (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'rnn/lstm_cell/bias':     (npr.rand(state_size * 4) - 0.5) * 0.15,
             'rnn/lstm_cell/w_f_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_i_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_o_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'linear_dihedrals/weights': (npr.rand(state_size, alphabet_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':  ((npr.rand(alphabet_size) - 0.5) * 0.05) + 0.2,
             'alphabet':                 (npr.rand(alphabet_size, output_size) - 0.5) * 2 * np.pi}

        # values sourced from autograd
        self._testCore(c_train, [c_eval], w,
                       {'all/loss':   [[5749.8842695544472], [1214.0236709015039], [4477.4536431732622]],
                        'drmsds': [[[6810.93141274,  5996.26703552,  6266.73531271,  5996.79034482,  8448.68989779,  7339.23759889,  6495.75602409,  3987.2847927 ,  3682.4609538 ,  5162.77609403,  7394.21053314,  8052.43790807,  6995.51878632,  5897.56225985,  2633.94578064,  6245.70370101,  8363.1074345 ,  6265.9235987 ,  5435.23899394,  4841.0296186 ,  3436.44231215,  5204.70090292,  7000.12182226,  5205.94677789,  4971.43059839,  1366.74051294]],
                                   [[1409.65443487,   912.96979174,  1324.95971502,  1155.05905596,  1873.02268022,  1461.09145851,  1276.67758657,   907.28835946,   845.15911505,  1099.01777833,  1362.75510935,  1652.93447853,  1435.89778189,  1290.01230389,   584.90055597,  1223.70175759,  1852.59314866,  1363.40375009,  1144.79580059,   980.70108435,   704.47227031,   931.66230177,  1486.44320628,  1021.24266313,  1112.5142134 ,  1151.68504191]],
                                   [[5365.37097742,  4620.97323775,  4919.09019944,  4695.76826843,  6698.20271211,  5752.4468811 ,  5073.7360612 ,  3113.26386125,  2849.20393335,  4049.33151433,  5783.61288749,  6365.89350299,  5499.509052  ,  4640.56363106,  1995.72129089,  4876.83136679,  6630.70311123,  4935.32653793,  4265.27589074,  3765.2461101 ,  2617.59618625,  4032.11265353,  5520.98966281,  4069.98934591,  3539.13583006,   737.90001636]]]})

    def testNonTrainableLinearAlphabetizedOutput(self):
        alphabet_size = 7

        c_train = deepcopy(c_train_template)
        c_train.architecture['tertiary_output'] = 'linear_alphabet'
        c_train.architecture['alphabet_size'] = alphabet_size
        c_train.architecture['alphabet_trainable'] = False

        c_eval = deepcopy(c_eval_template)
        c_eval.architecture['tertiary_output'] = 'linear_alphabet'
        c_eval.architecture['alphabet_size'] = alphabet_size
        c_eval.architecture['alphabet_trainable'] = False

        npr.seed(1)
        w = {'rnn/lstm_cell/kernel':    (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'rnn/lstm_cell/bias':     (npr.rand(state_size * 4) - 0.5) * 0.15,
             'rnn/lstm_cell/w_f_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_i_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_o_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'linear_dihedrals/weights': (npr.rand(state_size, alphabet_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':  ((npr.rand(alphabet_size) - 0.5) * 0.05) + 0.2,
             'alphabet':                 (npr.rand(alphabet_size, output_size) - 0.5) * 2 * np.pi}

        # values sourced from autograd
        self._testCore(c_train, [c_eval], w,
                       {'all/loss':   [[5749.8842814059763], [4612.4191128636667], [2938.0186556239128]],
                        'drmsds': [[[6810.93140889,  5996.26706865,  6266.73534689,  5996.79037183,  8448.68990083,  7339.23760914,  6495.75603058,  3987.28480556,  3682.46095596,  5162.7761008 ,  7394.21051847,  8052.43793065,  6995.51878223,  5897.56226507,  2633.94580308,  6245.70371783,  8363.10748073,  6265.92361545,  5435.23901599,  4841.0296248 ,  3436.44227842,  5204.70092673,  7000.12184112,  5205.94676574,  4971.43062402,  1366.74052709]],
                                   [[5519.92970725,  4766.83035411,  5061.85958019,  4833.93874721,  6883.75876644,  5921.63563204,  5225.07497515,  3205.88732686,  2936.83085503,  4166.74974989,  5955.37074939,  6545.99365902,  5657.70077364,  4773.55795702,  2062.55215377,  5022.60218232,  6813.89580758,  5076.19825577,  4390.0925107 ,  3878.6006903 ,  2703.61407999,  4157.26609015,  5678.14842935,  4190.15783905,  3690.61004153,   804.0400207]],
                                   [[3599.38172974,  2936.44102507,  3277.51469927,  3103.96812708,  4552.27045547,  3816.19778525,  3340.82378098,  2062.41185978,  1853.03338901,  2693.96566888,  3811.31643457,  4294.81985968,  3669.01640659,  3109.70849391,  1239.40868596,  3209.8947435 ,  4505.01492405,  3312.91222147,  2842.79957574,  2460.10093665,  1637.47747544,  2605.53445311,  3712.39956002,  2682.96569743,  1831.17418064,   227.93287694]]]})

    def testTrainableAngularAlphabetizedOutput(self):
        alphabet_size = 7

        c_train = deepcopy(c_train_template)
        c_train.architecture['tertiary_output'] = 'angular_alphabet'
        c_train.architecture['alphabet_size'] = alphabet_size

        c_eval = deepcopy(c_eval_template)
        c_eval.architecture['tertiary_output'] = 'angular_alphabet'
        c_eval.architecture['alphabet_size'] = alphabet_size

        npr.seed(1)
        w = {'rnn/lstm_cell/kernel':    (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'rnn/lstm_cell/bias':     (npr.rand(state_size * 4) - 0.5) * 0.15,
             'rnn/lstm_cell/w_f_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_i_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_o_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'linear_dihedrals/weights': (npr.rand(state_size, alphabet_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':  ((npr.rand(alphabet_size) - 0.5) * 0.05) + 0.2,
             'alphabet':                 (npr.rand(alphabet_size, output_size) - 0.5) * 2 * np.pi}

        # values sourced from autograd
        self._testCore(c_train, [c_eval], w,
                       {'all/loss':   [[8491.3773654441902], [1281.7034366501621], [1183.3535380281851]],
                        'drmsds': [[[ 9916.80147786,   8951.21693885, 9171.71529387,  8793.8169752 ,  12210.89968445, 10751.1247297 , 9557.96664572,   5880.63443573,  5494.98361096, 7565.5301144 ,  10851.14700973, 11668.37451299, 10214.71735466, 8608.40318419,   4025.50599477, 9191.97719289,  12086.85739522, 9132.79020536,  7955.41529632,  7164.65504524,   5215.31033356,  7728.12106495, 10181.10464591, 7652.21740348,  8069.29523878,  2735.22971677]],
                                   [[ 1519.91886204,   998.62876623,  1411.93108329,  1248.17088209,  2010.86567245,  1572.64353406,  1367.53907736,   940.51934755,   860.95856229,  1166.05451011,  1484.14498726,  1796.32792899,  1542.64286596,  1370.25449716,   565.71980942,  1310.06701004,  1989.76443743,  1451.89653837,  1219.23682396,  1032.07574481,   704.12663468,   997.60282064,  1596.34012292,  1094.87922418,  1007.56662422,  1064.41298539]],
                                   [[ 1379.77700506,   885.97105742,  1295.92587333,  1129.5501944 ,  1838.58031867,  1430.82835298,  1247.41840618,   873.43595292,   808.89961586,  1070.55534319,  1334.00722341,  1622.3667254 ,  1406.35893613,  1260.82120689,   533.93315958,  1194.34086895,  1820.54482917,  1332.21860356,  1116.24698385,   949.42068994,   663.81919489,   904.0619315 ,  1456.82662476,   997.42334098,  1100.69736628,  1113.16218343]]]},
                        rtol=5e-2, atol=5e-2)

    def testNonTrainableFixedAlphabetizedOutput(self):            
        alphabet_size = 60

        c_train = deepcopy(c_train_template)
        c_train.architecture['tertiary_output'] = 'linear_alphabet'
        c_train.architecture['alphabet_trainable'] = False
        c_train.io['alphabet_file'] = os.path.join(alphabets_dir, 'alphabetPointsA12CV3.csv')
        c_train.io['data_files'] = [os.path.join(train_dir, file) for file in train_files]

        c_eval = deepcopy(c_train)
        c_eval.io['data_files']                  = c_eval_template.io['data_files']
        c_eval.optimization['batch_size']        = c_eval_template.optimization['batch_size']

        npr.seed(1)
        w = {'rnn/lstm_cell/kernel':    (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'rnn/lstm_cell/bias':     (npr.rand(state_size * 4) - 0.5) * 0.15,
             'rnn/lstm_cell/w_f_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_i_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_o_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'linear_dihedrals/weights': (npr.rand(state_size, alphabet_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':  ((npr.rand(alphabet_size) - 0.5) * 0.05) + 0.2}

        # values sourced from autograd
        self._testCore(c_train, [c_eval], w,
                       {'all/loss':   [[7131.1200421676858], [6065.267770659013], [4392.9599220732198]],
                        'drmsds': [[[8376.00400852,  7485.82907194,  7730.05596506,  7406.60680739, 10344.81900041,  9058.33890916,  8038.79459936,  4940.20136457,  4594.46247086,   6373.2182178 , 9136.51228842,  9875.47748482,  8617.77772632,  7263.23882965,  3334.14817804,  7729.94976305, 10239.62016289,  7710.11437913,  6704.62408207,  6011.38025063,  4332.11111965,  6476.35423903,  8603.08793306,  6438.64203381,  6531.23253428,  2056.51967642]],
                                   [[7168.72310193,  6336.3398706 ,  6600.62784981,  6318.66208194,  8880.7160074 ,  7731.65960749,  6848.24911014,  4205.03522376,  3890.94813993,  5439.07531749,  7791.72471636,  8468.75993537,  7365.4514001 ,  6209.1275917 ,  2794.69930346,  6584.5371745 ,  8790.23570887,  6595.3204659 ,  5725.40751283,  5108.19709995,  3641.55799688,  5495.59637478,  7365.96167594,  5487.27567313,  5326.93671762,  1526.13637924]],
                                   [[5270.17638262,  4528.77656031,  4828.83458118,  4608.80397653,  6578.78169561,  5646.83721723,  4979.08370178,  3056.97803117,  2795.2679541 ,  3975.1776004 ,  5675.70032556,  6253.09166031,  5398.10432121,  4556.06349111,  1956.40739258,  4786.16372001,  6511.58390578,  4846.10167266,  4188.81123895,  3694.15025071,  2565.11123429,  3955.97647311,  5421.83749364,  3994.14066761,  3443.46129087,   701.53513458]]]})

    def testTrainableFixedAlphabetizedOutput(self):            
        alphabet_size = 60

        c_train = deepcopy(c_train_template)
        c_train.architecture['tertiary_output'] = 'linear_alphabet'
        c_train.io['alphabet_file'] = os.path.join(alphabets_dir, 'alphabetPointsA12CV3.csv')
        c_train.io['data_files'] = [os.path.join(train_dir, file) for file in train_files]

        c_eval = deepcopy(c_train)
        c_eval.io['data_files']           = c_eval_template.io['data_files']
        c_eval.optimization['batch_size'] = c_eval_template.optimization['batch_size']

        npr.seed(1)
        w = {'rnn/lstm_cell/kernel':    (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'rnn/lstm_cell/bias':     (npr.rand(state_size * 4) - 0.5) * 0.15,
             'rnn/lstm_cell/w_f_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_i_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_o_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'linear_dihedrals/weights': (npr.rand(state_size, alphabet_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':  ((npr.rand(alphabet_size) - 0.5) * 0.05) + 0.2}

        # values sourced from initial version of TF code as there's no autograd equivalent for this.
        self._testCore(c_train, [c_eval], w,
                       {'all/loss':   [[7131.1162], [5772.2246], [3154.8499]],
                        'drmsds': [[[8376.0078125 ,  7485.83154297,  7730.04833984,  7406.60400391,  10344.8046875,  9058.32714844,  8038.79150391,  4940.19970703,   4594.4609375 ,   6373.21630859,   9136.5078125 ,   9875.46679688,   8617.76660156,   7263.24121094,   3334.14746094,   7729.94921875,  10239.60351562,   7710.10986328,   6704.62353516,   6011.37939453,   4332.11376953,   6476.3515625 ,   8603.07910156,   6438.63916016,   6531.22802734,   2056.52001953]],
                                   [[6836.27783203,  6020.08056641,  6290.14794922,  6019.4375    ,  8478.03125   ,  7366.54150391,  6520.77734375,  4003.28393555,  3698.11523438,  5182.43701172,  7421.57128906,  8081.32568359,  7021.09130859,  5919.48925781,  2646.87548828,  6269.45605469,  8391.78027344,  6288.86425781,  5456.07763672,  4860.10449219,  3452.1574707 ,  5225.75097656,  7025.61962891,  5225.86621094,  4996.19824219,  1380.49694824]],
                                   [[3852.5715332 ,  3179.64306641,  3512.83666992,  3333.74072266,  4859.81396484,  4093.08984375,  3589.01855469,  2212.39794922,  1994.98596191,  2889.07397461,  4094.46704102,  4592.90136719,  3932.20654297,  3329.4152832 ,  1347.35717773,  3448.63134766,  4810.43115234,  3545.81518555,  3047.08032227,  2647.66943359,  1777.6348877 ,  2810.90161133,  3972.15185547,  2883.67016602,  2068.24853516,   200.33422852]]]})

    def testAngleShift(self):
        c_train = deepcopy(c_train_template)
        c_train.initialization['angle_shift'] = [np.pi / 3., np.pi / 3., np.pi / 3.]

        c_eval = deepcopy(c_eval_template)
        c_eval.initialization['angle_shift'] = [np.pi / 3., np.pi / 3., np.pi / 3.]

        # values sourced from autograd
        self._testCore(c_train, [c_eval], w_template,
                       {'all/loss':   [[7130.749845383034], [1975.2657117215592], [9092.4642815397001]],
                        'drmsds': [[[ 8376.56603401,   7485.19954089,   7729.65566752,   7405.89560398,  10344.30571807,   9057.98635654,   8038.16113936,   4940.08556049,   4593.91047185,   6372.57517422,   9136.64994273,   9875.39340298,   8617.03828373,   7262.53895678,   3333.22083147,   7729.66940393,  10239.92945622,   7709.89812485,   6704.94210066,   6010.82462885,   4330.93034461,   6475.92476608,   8602.90659664,   6438.27164438,   6531.28841148,   2055.72781766]],
                                   [[2440.91351196,  1830.81197137,  2213.40907455,  2057.41630536,  3133.57369254,  2549.54558376,  2214.28480715,  1403.15517294,  1246.06894865,  1818.3245171 ,  2509.59949493,  2916.92699303,  2468.07727632,  2116.39010144,   798.8566617 ,  2126.8152878 ,  3100.67760812,  2258.05503026,  1922.28163659,  1627.05422116,  1044.91231739,  1682.08574637,  2525.50315609,  1778.3184391 ,  920.19946215,   653.65148693]],
                                   [[10597.64420766,   9597.03482145,   9808.38092234,   9405.94848818,  13033.98673639,  11498.18007662,  10229.14529953,   6298.46736627,   5894.99329773,   8093.37394682,  11607.18789949,  12458.61272031,  10918.9755659 ,   9202.1783646 ,   4333.69246952,   9837.59991849,  12901.53876542,   9761.04679194,   8508.599909  ,   7675.38099937,   5607.59225643,   8282.26513762,  10877.37931377,   8188.67657225,   8748.9487059 ,   3037.24076702]]]})

    def testHigherOrderBidirectionalMultilayers(self):
        # the numbers below are not based on an independent reference implementation, just the initial values I got for this test.
        # their primary purpose is to serve as a check for unintended changes in future releases of TF or this code base.

        c_train = deepcopy(c_train_template)
        c_train.architecture['recurrent_layer_size'] = [state_size, state_size]
        c_train.architecture['bidirectional'] = True
        c_train.architecture['higher_order_layers'] = True

        c_eval_wt_val = deepcopy(c_train)
        c_eval_wt_val.io['data_files'] = c_eval_template.io['data_files']
        c_eval_wt_val.optimization['batch_size'] = c_eval_template.optimization['batch_size']

        npr.seed(1)
        w = {'layer0/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer0/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer1/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand((2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer1/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand((2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,
               
             'linear_dihedrals/weights':                       (npr.rand(state_size * 2, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':                        ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

        # values sourced from autograd
        self._testCore(c_train, [c_eval_wt_val], w, {'all/loss': [[1111.9243], [1498.2306], [8704.0009], [5956.4995], [5283.3504]]})

    def testHigherOrderBidirectionalMultilayersWithAlphabetizedDihedralsInBetweenSingleAlphabet(self):
        # the numbers below are not based on an independent reference implementation, just the initial values I got for this test.
        # their primary purpose is to serve as a check for unintended changes in future releases of TF or this code base.

        alphabet_size = 7

        c_train = deepcopy(c_train_template)
        c_train.architecture['recurrent_layer_size'] = [state_size, state_size]
        c_train.architecture['tertiary_output'] = 'linear_alphabet'
        c_train.architecture['alphabet_size'] = alphabet_size
        c_train.architecture['bidirectional'] = True
        c_train.architecture['higher_order_layers'] = True
        c_train.architecture['include_dihedrals_between_layers'] = True

        c_eval_wt_val = deepcopy(c_train)
        c_eval_wt_val.io['data_files'] = c_eval_template.io['data_files']
        c_eval_wt_val.optimization['batch_size'] = c_eval_template.optimization['batch_size']

        npr.seed(1)
        w = {'layer0/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer0/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer0/linear_dihedrals/weights':                (npr.rand(state_size * 2, alphabet_size) - 0.5) * 0.05,
             'layer0/linear_dihedrals/biases':                 ((npr.rand(alphabet_size) - 0.5) * 0.05) + 0.2,
               
             'layer1/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(((2 * state_size) + output_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer1/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(((2 * state_size) + output_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,
                
             'linear_dihedrals/weights':                       (npr.rand(state_size * 2, alphabet_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':                        ((npr.rand(alphabet_size) - 0.5) * 0.05) + 0.2,
               
             'alphabet':                                       (npr.rand(alphabet_size, output_size) - 0.5) * 2 * np.pi}

        # values sourced from autograd
        self._testCore(c_train, [c_eval_wt_val], w, {'all/loss': [[2518.6995], [3579.6421], [7564.6065], [4977.0184], [1254.0698]]})

    def testHigherOrderBidirectionalMultilayersWithOnlyAlphabetizedDihedralsInBetweenSingleAlphabet(self):
        # the numbers below are not based on an independent reference implementation, just the initial values I got for this test.
        # their primary purpose is to serve as a check for unintended changes in future releases of TF or this code base.

        alphabet_size = 7

        c_train = deepcopy(c_train_template)
        c_train.architecture['recurrent_layer_size'] = [state_size, state_size]
        c_train.architecture['tertiary_output'] = 'linear_alphabet'
        c_train.architecture['alphabet_size'] = alphabet_size
        c_train.architecture['bidirectional'] = True
        c_train.architecture['higher_order_layers'] = True
        c_train.architecture['include_dihedrals_between_layers'] = True
        c_train.architecture['include_recurrent_outputs_between_layers'] = False

        c_eval_wt_val = deepcopy(c_train)
        c_eval_wt_val.io['data_files'] = c_eval_template.io['data_files']
        c_eval_wt_val.optimization['batch_size'] = c_eval_template.optimization['batch_size']

        npr.seed(1)
        w = {'layer0/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer0/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer0/linear_dihedrals/weights':                (npr.rand(state_size * 2, alphabet_size) - 0.5) * 0.05,
             'layer0/linear_dihedrals/biases':                 ((npr.rand(alphabet_size) - 0.5) * 0.05) + 0.2,
               
             'layer1/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(output_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer1/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(output_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,
                
             'linear_dihedrals/weights':                       (npr.rand(state_size * 2, alphabet_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':                        ((npr.rand(alphabet_size) - 0.5) * 0.05) + 0.2,
               
             'alphabet':                                       (npr.rand(alphabet_size, output_size) - 0.5) * 2 * np.pi}

        # values sourced from autograd
        self._testCore(c_train, [c_eval_wt_val], w, {'all/loss': [[2490.173], [7593.94], [958.31013], [6398.2544], [1689.9448]]})

    def testHigherOrderBidirectionalMultilayersWithAlphabetizedDihedralsInBetweenSeparateAlphabets(self):
        # the numbers below are not based on an independent reference implementation, just the initial values I got for this test.
        # their primary purpose is to serve as a check for unintended changes in future releases of TF or this code base.

        alphabet_size = [7, 13]

        c_train = deepcopy(c_train_template)
        c_train.architecture['recurrent_layer_size'] = [state_size, state_size]
        c_train.architecture['tertiary_output'] = 'linear_alphabet'
        c_train.architecture['alphabet_size'] = alphabet_size
        c_train.architecture['bidirectional'] = True
        c_train.architecture['higher_order_layers'] = True
        c_train.architecture['include_dihedrals_between_layers'] = True

        c_eval_wt_val = deepcopy(c_train)
        c_eval_wt_val.io['data_files'] = c_eval_template.io['data_files']
        c_eval_wt_val.optimization['batch_size'] = c_eval_template.optimization['batch_size']

        npr.seed(1)
        w = {'layer0/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer0/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer0/linear_dihedrals/weights':                (npr.rand(state_size * 2, alphabet_size[0]) - 0.5) * 0.05,
             'layer0/linear_dihedrals/biases':                 ((npr.rand(alphabet_size[0]) - 0.5) * 0.05) + 0.2,

             'layer0/alphabet':                                (npr.rand(alphabet_size[0], output_size) - 0.5) * 2 * np.pi,
               
             'layer1/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(((2 * state_size) + output_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer1/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(((2 * state_size) + output_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,
                
             'linear_dihedrals/weights':                       (npr.rand(state_size * 2, alphabet_size[1]) - 0.5) * 0.05,
             'linear_dihedrals/biases':                        ((npr.rand(alphabet_size[1]) - 0.5) * 0.05) + 0.2,
               
             'alphabet':                                       (npr.rand(alphabet_size[1], output_size) - 0.5) * 2 * np.pi}

        # values sourced from autograd
        self._testCore(c_train, [c_eval_wt_val], w, {'all/loss': [[8544.1307], [984.68151], [5165.786], [1092.0468], [5219.408]]})

    def testHigherOrderBidirectionalMultilayersWithDihedralsInBetween(self):
        # the numbers below are not based on an independent reference implementation, just the initial values I got for this test.
        # their primary purpose is to serve as a check for unintended changes in future releases of TF or this code base.

        c_train = deepcopy(c_train_template)
        c_train.architecture['recurrent_layer_size'] = [state_size, state_size]
        c_train.architecture['bidirectional'] = True
        c_train.architecture['higher_order_layers'] = True
        c_train.architecture['include_dihedrals_between_layers'] = True

        c_eval_wt_val = deepcopy(c_train)
        c_eval_wt_val.io['data_files'] = c_eval_template.io['data_files']
        c_eval_wt_val.optimization['batch_size'] = c_eval_template.optimization['batch_size']

        npr.seed(1)
        w = {'layer0/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer0/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer0/linear_dihedrals/weights':                (npr.rand(state_size * 2, output_size) - 0.5) * 0.05,
             'layer0/linear_dihedrals/biases':                 ((npr.rand(output_size) - 0.5) * 0.05) + 0.2,
               
             'layer1/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(((2 * state_size) + output_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer1/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(((2 * state_size) + output_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,
                
             'linear_dihedrals/weights':                       (npr.rand(state_size * 2, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':                        ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

        # values sourced from autograd
        self._testCore(c_train, [c_eval_wt_val], w, {'all/loss': [[1074.5903], [1161.7908], [8912.2383], [1167.9236], [7854.7287]]})

    def testAlphabetTemperature(self):
        alphabet_size = 7

        c_train = deepcopy(c_train_template)
        c_train.architecture['tertiary_output'] = 'linear_alphabet'
        c_train.architecture['alphabet_size'] = alphabet_size
        c_train.optimization['alphabet_temperature'] = 0.01

        c_eval = deepcopy(c_eval_template)
        c_eval.architecture['tertiary_output'] = 'linear_alphabet'
        c_eval.architecture['alphabet_size'] = alphabet_size
        c_eval.optimization['alphabet_temperature'] = 0.01

        npr.seed(1)
        w = {'rnn/lstm_cell/kernel':    (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'rnn/lstm_cell/bias':     (npr.rand(state_size * 4) - 0.5) * 0.15,
             'rnn/lstm_cell/w_f_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_i_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_o_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'linear_dihedrals/weights': (npr.rand(state_size, alphabet_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':  ((npr.rand(alphabet_size) - 0.5) * 0.05) + 0.2,
             'alphabet':                 (npr.rand(alphabet_size, output_size) - 0.5) * 2 * np.pi}

        self._testCore(c_train, [c_eval], w, {'all/loss': [[5081.0432], [5074.5049], [8391.9434], [5796.6553], [1982.2956]]})

    def testTwoLayersResidualsEveryLayer(self):
        # the numbers below are not based on an independent reference implementation, just the initial values I got for this test.
        # their primary purpose is to serve as a check for unintended changes in future releases of TF or this code base.

        c_train = deepcopy(c_train_template)
        c_train.architecture['recurrent_layer_size'] = [state_size, state_size]
        c_train.architecture['bidirectional'] = True
        c_train.architecture['higher_order_layers'] = True
        c_train.architecture['residual_connections_every_n_layers'] = 1

        c_eval_wt_val = deepcopy(c_train)
        c_eval_wt_val.io['data_files'] = c_eval_template.io['data_files']
        c_eval_wt_val.optimization['batch_size'] = c_eval_template.optimization['batch_size']

        npr.seed(1)
        w = {'layer0/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer0/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer1/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand((2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer1/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand((2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,
               
             'linear_dihedrals/weights':                       (npr.rand(state_size * 2, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':                        ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

        # values sourced from autograd
        self._testCore(c_train, [c_eval_wt_val], w, {'all/loss': [[1112.1163], [1538.6641], [7401.3466], [1555.9832], [2760.1791]]})

    def testThreeLayersResidualsEveryLayerStartSecondLayer(self):
        # the numbers below are not based on an independent reference implementation, just the initial values I got for this test.
        # their primary purpose is to serve as a check for unintended changes in future releases of TF or this code base.

        c_train = deepcopy(c_train_template)
        c_train.architecture['recurrent_layer_size'] = [state_size, state_size, state_size]
        c_train.architecture['bidirectional'] = True
        c_train.architecture['higher_order_layers'] = True
        c_train.architecture['residual_connections_every_n_layers'] = 1
        c_train.architecture['first_residual_connection_from_nth_layer'] = 2

        c_eval_wt_val = deepcopy(c_train)
        c_eval_wt_val.io['data_files'] = c_eval_template.io['data_files']
        c_eval_wt_val.optimization['batch_size'] = c_eval_template.optimization['batch_size']

        npr.seed(1)
        w = {'layer0/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer0/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer1/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand((2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer1/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand((2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer2/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand((2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer2/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer2/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand((2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer2/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,
            
             'linear_dihedrals/weights':                       (npr.rand(state_size * 2, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':                        ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

        # values sourced from autograd
        self._testCore(c_train, [c_eval_wt_val], w, {'all/loss': [[1204.4078], [4708.3481], [2588.7672], [8325.1083]]})

    def testFourLayersResidualsEverySecondLayerStartFirstLayer(self):
        # the numbers below are not based on an independent reference implementation, just the initial values I got for this test.
        # their primary purpose is to serve as a check for unintended changes in future releases of TF or this code base.

        c_train = deepcopy(c_train_template)
        c_train.architecture['recurrent_layer_size'] = [state_size, state_size, state_size, state_size]
        c_train.architecture['bidirectional'] = True
        c_train.architecture['higher_order_layers'] = True
        c_train.architecture['residual_connections_every_n_layers'] = 2

        c_eval_wt_val = deepcopy(c_train)
        c_eval_wt_val.io['data_files'] = c_eval_template.io['data_files']
        c_eval_wt_val.optimization['batch_size'] = c_eval_template.optimization['batch_size']

        npr.seed(1)
        w = {'layer0/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer0/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer1/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand((2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer1/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand((2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer2/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand((2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer2/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer2/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand((2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer2/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer3/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand((2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer3/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer3/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand((2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer3/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,
            
             'linear_dihedrals/weights':                       (npr.rand(state_size * 2, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':                        ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

        # values sourced from autograd
        self._testCore(c_train, [c_eval_wt_val], w, {'all/loss': [[1134.7569], [2224.3132], [8698.0324], [4956.8409], [4446.7251]]})

    def testFourLayersResidualsEverySecondLayerStartSecondLayer(self):
        # the numbers below are not based on an independent reference implementation, just the initial values I got for this test.
        # their primary purpose is to serve as a check for unintended changes in future releases of TF or this code base.

        c_train = deepcopy(c_train_template)
        c_train.architecture['recurrent_layer_size'] = [state_size, state_size, state_size, state_size]
        c_train.architecture['bidirectional'] = True
        c_train.architecture['higher_order_layers'] = True
        c_train.architecture['residual_connections_every_n_layers'] = 2
        c_train.architecture['first_residual_connection_from_nth_layer'] = 2

        c_eval_wt_val = deepcopy(c_train)
        c_eval_wt_val.io['data_files'] = c_eval_template.io['data_files']
        c_eval_wt_val.optimization['batch_size'] = c_eval_template.optimization['batch_size']

        npr.seed(1)
        w = {'layer0/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer0/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer1/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand((2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer1/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand((2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer2/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand((2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer2/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer2/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand((2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer2/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer3/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand((2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer3/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer3/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand((2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer3/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,
            
             'linear_dihedrals/weights':                       (npr.rand(state_size * 2, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':                        ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

        # values sourced from autograd
        self._testCore(c_train, [c_eval_wt_val], w, {'all/loss': [[1139.4403], [2549.3982], [7044.8929], [1352.3712]]})

    def testTwoLayersSkipsFromRecurrentToOutput(self):
        # the numbers below are not based on an independent reference implementation, just the initial values I got for this test.
        # their primary purpose is to serve as a check for unintended changes in future releases of TF or this code base.

        c_train = deepcopy(c_train_template)
        c_train.architecture['recurrent_layer_size'] = [state_size, state_size]
        c_train.architecture['bidirectional'] = True
        c_train.architecture['higher_order_layers'] = True
        c_train.architecture['recurrent_to_output_skip_connections'] = True

        c_eval_wt_val = deepcopy(c_train)
        c_eval_wt_val.io['data_files'] = c_eval_template.io['data_files']
        c_eval_wt_val.optimization['batch_size'] = c_eval_template.optimization['batch_size']

        npr.seed(1)
        w = {'layer0/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer0/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer1/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand((2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer1/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand((2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,
               
             'linear_dihedrals/weights':                       (npr.rand(state_size * 2 * 2, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':                        ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

        # values sourced from autograd
        self._testCore(c_train, [c_eval_wt_val], w, {'all/loss': [[1154.8752], [3163.6017], [7615.6151], [2933.5653], [1573.9504]]})

    def testTwoLayersSkipsFromInputToRecurrent(self):
        # the numbers below are not based on an independent reference implementation, just the initial values I got for this test.
        # their primary purpose is to serve as a check for unintended changes in future releases of TF or this code base.

        c_train = deepcopy(c_train_template)
        c_train.architecture['recurrent_layer_size'] = [state_size, state_size]
        c_train.architecture['bidirectional'] = True
        c_train.architecture['higher_order_layers'] = True
        c_train.architecture['input_to_recurrent_skip_connections'] = True

        c_eval_wt_val = deepcopy(c_train)
        c_eval_wt_val.io['data_files'] = c_eval_template.io['data_files']
        c_eval_wt_val.optimization['batch_size'] = c_eval_template.optimization['batch_size']

        npr.seed(1)
        w = {'layer0/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer0/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer1/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(input_size + (2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer1/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(input_size + (2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,
               
             'linear_dihedrals/weights':                       (npr.rand(state_size * 2, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':                        ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

        # values sourced from autograd
        self._testCore(c_train, [c_eval_wt_val], w, {'all/loss': [[1172.2696], [3371.7709], [6487.6198], [3878.7231]]})

    def testFourLayersSkipsFromAllToRecurrent(self):
        # the numbers below are not based on an independent reference implementation, just the initial values I got for this test.
        # their primary purpose is to serve as a check for unintended changes in future releases of TF or this code base.

        c_train = deepcopy(c_train_template)
        c_train.architecture['recurrent_layer_size'] = [state_size, state_size, state_size, state_size]
        c_train.architecture['bidirectional'] = True
        c_train.architecture['higher_order_layers'] = True
        c_train.architecture['all_to_recurrent_skip_connections'] = True

        c_eval_wt_val = deepcopy(c_train)
        c_eval_wt_val.io['data_files'] = c_eval_template.io['data_files']
        c_eval_wt_val.optimization['batch_size'] = c_eval_template.optimization['batch_size']

        npr.seed(1)
        w = {'layer0/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer0/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer1/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(input_size + (2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer1/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(input_size + (2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer2/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(input_size + 2 * (2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer2/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer2/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(input_size + 2 * (2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer2/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,
            
             'layer3/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(input_size + 3 * (2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer3/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer3/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(input_size + 3 * (2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer3/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,
            
             'linear_dihedrals/weights':                       (npr.rand(state_size * 2, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':                        ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

        # values sourced from autograd
        self._testCore(c_train, [c_eval_wt_val], w, {'all/loss': [[1180.7234], [3626.7376], [7490.6593], [951.0652], [5202.779]]})

    def testFourLayersResidualsEverySecondLayerStartSecondLayerAndSkipsFromAllToRecurrentAndRecurrentToOutput(self):
        # the numbers below are not based on an independent reference implementation, just the initial values I got for this test.
        # their primary purpose is to serve as a check for unintended changes in future releases of TF or this code base.

        c_train = deepcopy(c_train_template)
        c_train.architecture['recurrent_layer_size'] = [state_size, state_size, state_size, state_size]
        c_train.architecture['bidirectional'] = True
        c_train.architecture['higher_order_layers'] = True
        c_train.architecture['all_to_recurrent_skip_connections'] = True
        c_train.architecture['recurrent_to_output_skip_connections'] = True
        c_train.architecture['residual_connections_every_n_layers'] = 2
        c_train.architecture['first_residual_connection_from_nth_layer'] = 2

        c_eval_wt_val = deepcopy(c_train)
        c_eval_wt_val.io['data_files'] = c_eval_template.io['data_files']
        c_eval_wt_val.optimization['batch_size'] = c_eval_template.optimization['batch_size']

        npr.seed(1)
        w = {'layer0/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer0/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer0/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer1/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(input_size + (2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer1/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(input_size + (2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer1/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer2/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(input_size + 2 * (2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer2/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer2/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(input_size + 2 * (2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer2/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer2/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,
            
             'layer3/bidirectional_rnn/fw/lstm_cell/kernel':  (npr.rand(input_size + 3 * (2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/fw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer3/bidirectional_rnn/fw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/fw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/fw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'layer3/bidirectional_rnn/bw/lstm_cell/kernel':  (npr.rand(input_size + 3 * (2 * state_size) + state_size, state_size * 4) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/bw/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'layer3/bidirectional_rnn/bw/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/bw/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'layer3/bidirectional_rnn/bw/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,
            
             'linear_dihedrals/weights':          (npr.rand(state_size * 2 * 4, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':           ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

        # values sourced from autograd
        self._testCore(c_train, [c_eval_wt_val], w, {'all/loss': [[1168.4316], [4341.5478], [1000.3198], [9159.6611], [9078.9536]]})

    def testNonlinearOutputProjection(self):
        nonlinear_out_proj_size = 8

        c_train = deepcopy(c_train_template)
        c_train.architecture['recurrent_nonlinear_out_proj_size'] = [nonlinear_out_proj_size]

        c_eval = deepcopy(c_eval_template)
        c_eval.architecture['recurrent_nonlinear_out_proj_size'] = [nonlinear_out_proj_size]

        npr.seed(1)
        w = {'rnn/lstm_cell/kernel':  (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'rnn/lstm_cell/bias':   (npr.rand(state_size * 4) - 0.5) * 0.15,
             'rnn/lstm_cell/w_f_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_i_diag': (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_o_diag': (npr.rand(state_size) - 0.5) * 0.05,

             'nonlinear_dihedrals_0/weights': (npr.rand(state_size, nonlinear_out_proj_size) - 0.5) * 0.05,
             'nonlinear_dihedrals_0/biases':  ((npr.rand(nonlinear_out_proj_size) - 0.5) * 0.05) + 0.2,
            
             'linear_dihedrals/weights': (npr.rand(nonlinear_out_proj_size, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':  ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

        # values sourced from autograd
        self._testCore(c_train, [c_eval], w, {'all/loss': [[1161.2987], [4173.3604], [1835.3266], [9014.5103], [6143.861]]})

    def testResidueMasking(self):
        # values sourced from manual Mathematica comparison (for very initial evaluation step)
        train_dir = base_dir + 'data/unofficial/tfrecord/training3_with_random_masks/'
        eval_dir = base_dir + 'data/unofficial/tfrecord/test3_with_random_masks/'

        c_train = deepcopy(c_train_template)
        c_train.io['data_files'] = [os.path.join(train_dir, file) for file in train_files]

        c_eval = deepcopy(c_eval_template)
        c_eval.io['data_files'] = [os.path.join(eval_dir, file) for file in eval_files]

        self._testCore(c_train, [c_eval], w_template,
                       {'all/loss':   [[1113.3933], [1290.5278], [8462.5641]],
                        'drmsds': [[[1311.66455078,   755.68762207,  1223.95471191,  1101.61340332, 1751.07995605,  1260.42272949,  1115.22094727,   822.32305908,  838.7767334 ,   988.93121338,  1217.94262695,  1400.19067383,  1290.76586914,  1189.64904785,   614.68103027,  1089.93945312,  1655.42382812,  1200.71838379,  1012.00054932,   868.31262207,  626.87713623,   848.3427124 ,  1326.40600586,   923.28723145,  1277.94665527,  1236.06921387]],
                                   [[1592.76782227,   936.25463867,  1434.36621094,  1355.26599121, 2128.32958984,  1507.71264648,  1334.89501953,   921.93353271,  871.11035156,  1162.04870605,  1526.84118652,  1744.01037598,  1583.15515137,  1395.01635742,   608.64050293,  1297.68115234,  2005.31347656,  1419.97167969,  1181.16638184,   994.47711182,  692.53173828,  1031.44995117,  1601.7322998 ,  1115.30993652,  1023.69152832,  1088.04785156]],
                                   [[10292.45605469,  8507.06347656, 9141.60253906,  9164.97265625, 12755.04589844, 10312.52246094, 9240.44140625,   5798.40185547,  5179.17822266,   7526.60400391,  11017.83496094,  11359.52050781,  10519.12988281,   8670.64355469,   4119.57373047,   9012.40820312,  12090.49121094,   8942.54394531,   7627.59033203,   6814.18896484,  5588.13134766,   7940.984375  ,  10130.20410156,   7740.75097656,  7889.84570312,   2644.55126953]]]})

    def testResidueMaskingWithZeroResidueShift(self):
        # values sourced from manual comparison in Jupyter
        train_dir = base_dir + 'data/CASP11Thinning30EvoUniParcBakerJackHMMERNeg10JackHMMERNeg10/training/full/'
        eval_dir = base_dir + '/data/CASP11Thinning30EvoUniParcBakerJackHMMERNeg10JackHMMERNeg10/validation/sample/'
        train_files = ['1', '2', '3']
        eval_files = ['1']
        state_size = 2

        c_train = deepcopy(c_train_template)
        c_train.io['data_files'] = [os.path.join(train_dir, file) for file in train_files]
        c_train.io['num_evo_entries'] = 42
        c_train.io['num_edge_residues'] = 0
        c_train.architecture['recurrent_layer_size'] = [state_size]
        c_train.loss['tertiary_normalization'] = 'first'
        c_train.optimization['batch_size'] = 224
        c_train.optimization['num_steps'] = 700

        c_eval = deepcopy(c_train)
        c_eval.io['data_files'] = [os.path.join(eval_dir, file) for file in eval_files]
        c_eval.io['evaluation_sub_groups'] = ['10', '20', '30', '40', '50', '70', '90']
        c_eval.optimization['min_after_dequeue'] = 1

        npr.seed(1)
        w = {'rnn/lstm_cell/kernel':    (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'rnn/lstm_cell/bias':     (npr.rand(state_size * 4) - 0.5) * 0.15,
             'rnn/lstm_cell/w_f_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_i_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_o_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'linear_dihedrals/weights': (npr.rand(state_size, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':  ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

        self._testCore(c_train, [c_eval], w, {'all/loss': [[6097.5014], [37162.122], [32348.877]],
                                              '10/loss':  [[4512.7651], [27956.369], [24301.126]],
                                              '20/loss':  [[5210.0796], [32712.424], [28429.001]],
                                              '30/loss':  [[5652.4761], [35114.963], [30551.239]],
                                              '40/loss':  [[5077.8084], [31967.755], [27801.285]],
                                              '50/loss':  [[6515.1100], [39263.562], [34194.144]],
                                              '70/loss':  [[7503.8086], [44695.819], [38947.882]],
                                              '90/loss':  [[7127.6367], [42440.350], [36978.729]]})

    def testBasicCudnnLSTM(self):
        cudnnLSTM_state_size = 2220

        c_train = deepcopy(c_train_template)
        c_train.architecture['recurrent_unit'] = 'CudnnLSTM'

        c_eval = deepcopy(c_eval_template)
        c_eval.architecture['recurrent_unit'] = 'CudnnLSTM'

        npr.seed(1)
        w = {'fw/cudnn_lstm/opaque_kernel': (npr.rand(cudnnLSTM_state_size) - 0.5) * 0.05,

             'linear_dihedrals/weights': (npr.rand(state_size, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':  ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

        self._testCore(c_train, [c_eval], w, {'all/loss': [[1135.6718], [2027.7658], [9087.0293], [8864.8125], [4934.1099]]},
                       rtol=1e-4, atol=1e-4, use_gpu=True, restart_every_iteration=True)

    def testCudnnLSTM(self):
        cudnnLSTM_state_size = 4740

        train_dir = base_dir + 'data/CASP11Thinning30TwoResidueShiftEvoUniParcBakerJackHMMERNeg10JackHMMERNeg10/training/full/'
        eval_dir = base_dir + '/data/CASP11Thinning30TwoResidueShiftEvoUniParcBakerJackHMMERNeg10JackHMMERNeg10/validation/sample/'
        train_files = ['1', '2', '3']
        eval_files = ['1']

        c_train = deepcopy(c_train_template)
        c_train.io['data_files'] = [os.path.join(train_dir, file) for file in train_files]
        c_train.io['num_evo_entries'] = 42
        c_train.architecture['recurrent_unit'] = 'CudnnLSTM'
        c_train.architecture['include_evolutionary'] = True

        c_eval = deepcopy(c_train)
        c_eval.io['data_files'] = [os.path.join(eval_dir, file) for file in eval_files]
        c_eval.optimization['batch_size'] = 52
        c_eval.queueing['min_after_dequeue'] = 1

        npr.seed(1)
        w = {'fw/cudnn_lstm/opaque_kernel': (npr.rand(cudnnLSTM_state_size) - 0.5) * 0.05,

             'linear_dihedrals/weights': (npr.rand(state_size, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':  ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

        self._testCore(c_train, [c_eval], w, {'all/loss': [[1142.7958], [2685.1274], [4247.5918], [1318.0102], [3206.9534]]},
                       rtol=1e-4, atol=1e-4, use_gpu=True, restart_every_iteration=True)

    def testBidirectionalCudnnLSTM(self):
        cudnnLSTM_state_size = 4740

        train_dir = base_dir + 'data/CASP11Thinning30TwoResidueShiftEvoUniParcBakerJackHMMERNeg10JackHMMERNeg10/training/full/'
        eval_dir = base_dir + '/data/CASP11Thinning30TwoResidueShiftEvoUniParcBakerJackHMMERNeg10JackHMMERNeg10/validation/sample/'
        train_files = ['1', '2', '3']
        eval_files = ['1']

        c_train = deepcopy(c_train_template)
        c_train.io['data_files'] = [os.path.join(train_dir, file) for file in train_files]
        c_train.io['num_evo_entries'] = 42
        c_train.architecture['recurrent_unit'] = 'CudnnLSTM'
        c_train.architecture['include_evolutionary'] = True
        c_train.architecture['bidirectional'] = True

        c_eval = deepcopy(c_train)
        c_eval.io['data_files'] = [os.path.join(eval_dir, file) for file in eval_files]
        c_eval.optimization['batch_size'] = 52
        c_eval.queueing['min_after_dequeue'] = 1

        npr.seed(1)
        w = {'fw/cudnn_lstm/opaque_kernel': (npr.rand(cudnnLSTM_state_size) - 0.5) * 0.05,
             'bw/cudnn_lstm/opaque_kernel': (npr.rand(cudnnLSTM_state_size) - 0.5) * 0.05,

             'linear_dihedrals/weights': (npr.rand(state_size * 2, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':  ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

        self._testCore(c_train, [c_eval], w, {'all/loss': [[1137.2385], [2556.5674], [4345.2259], [1203.1003], [2087.2541]]},
                       rtol=1e-4, atol=1e-4, use_gpu=True, restart_every_iteration=True)

    def testTwoLOLayeredBidirectionalCudnnLSTM(self):
        cudnnLSTM_state_size = 6660

        train_dir = base_dir + 'data/CASP11Thinning30TwoResidueShiftEvoUniParcBakerJackHMMERNeg10JackHMMERNeg10/training/full/'
        eval_dir = base_dir + '/data/CASP11Thinning30TwoResidueShiftEvoUniParcBakerJackHMMERNeg10JackHMMERNeg10/validation/sample/'
        train_files = ['1', '2', '3']
        eval_files = ['1']

        c_train = deepcopy(c_train_template)
        c_train.io['data_files'] = [os.path.join(train_dir, file) for file in train_files]
        c_train.io['num_evo_entries'] = 42
        c_train.architecture['recurrent_unit'] = 'CudnnLSTM'
        c_train.architecture['include_evolutionary'] = True
        c_train.architecture['bidirectional'] = True
        c_train.architecture['recurrent_layer_size'] = [state_size] * 2

        c_eval = deepcopy(c_train)
        c_eval.io['data_files'] = [os.path.join(eval_dir, file) for file in eval_files]
        c_eval.optimization['batch_size'] = 52
        c_eval.queueing['min_after_dequeue'] = 1

        npr.seed(1)
        w = {'fw/cudnn_lstm/opaque_kernel': (npr.rand(cudnnLSTM_state_size) - 0.5) * 0.1,
             'bw/cudnn_lstm/opaque_kernel': (npr.rand(cudnnLSTM_state_size) - 0.5) * 0.1,

             'linear_dihedrals/weights': (npr.rand(state_size * 2, output_size) - 0.5) * 0.1,
             'linear_dihedrals/biases':  ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

        self._testCore(c_train, [c_eval], w, {'all/loss': [[1144.789], [2778.5601], [4170.4731], [1512.8389], [4059.8572]]},
                       rtol=1e-4, atol=1e-4, use_gpu=True, restart_every_iteration=True)

    def testTwoHOLayeredBidirectionalCudnnLSTM(self):
        cudnnLSTM_0_state_size = 4740
        cudnnLSTM_1_state_size = 2820

        train_dir = base_dir + 'data/CASP11Thinning30TwoResidueShiftEvoUniParcBakerJackHMMERNeg10JackHMMERNeg10/training/full/'
        eval_dir = base_dir + '/data/CASP11Thinning30TwoResidueShiftEvoUniParcBakerJackHMMERNeg10JackHMMERNeg10/validation/sample/'
        train_files = ['1', '2', '3']
        eval_files = ['1']

        c_train = deepcopy(c_train_template)
        c_train.io['data_files'] = [os.path.join(train_dir, file) for file in train_files]
        c_train.io['num_evo_entries'] = 42
        c_train.architecture['recurrent_unit'] = 'CudnnLSTM'
        c_train.architecture['higher_order_layers'] = True
        c_train.architecture['include_evolutionary'] = True
        c_train.architecture['bidirectional'] = True
        c_train.architecture['recurrent_layer_size'] = [state_size] * 2

        c_eval = deepcopy(c_train)
        c_eval.io['data_files'] = [os.path.join(eval_dir, file) for file in eval_files]
        c_eval.optimization['batch_size'] = 52
        c_eval.queueing['min_after_dequeue'] = 1

        npr.seed(1)
        w = {'layer0/fw/cudnn_lstm/opaque_kernel': (npr.rand(cudnnLSTM_0_state_size) - 0.5) * 0.1,
             'layer0/bw/cudnn_lstm/opaque_kernel': (npr.rand(cudnnLSTM_0_state_size) - 0.5) * 0.1,
             'layer1/fw/cudnn_lstm/opaque_kernel': (npr.rand(cudnnLSTM_1_state_size) - 0.5) * 0.1,
             'layer1/bw/cudnn_lstm/opaque_kernel': (npr.rand(cudnnLSTM_1_state_size) - 0.5) * 0.1,

             'linear_dihedrals/weights': (npr.rand(state_size * 2, output_size) - 0.5) * 0.1,
             'linear_dihedrals/biases':  ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

        self._testCore(c_train, [c_eval], w, {'all/loss': [[1117.6941], [1791.2733], [4048.428], [1752.7554], [4215.1085]]},
                       rtol=1e-4, atol=1e-4, use_gpu=True, restart_every_iteration=True)

    def testEvaluationSubgroupsZerothOrderLoss(self):
        # values assumed correct because sum of subgroups adds up to total
        train_files = ['1', '2', '3']
        eval_files  = ['1']
        train_dir = base_dir + 'data/unofficial/tfrecord/training_long_with_subgroups/'
        eval_dir  = base_dir + 'data/unofficial/tfrecord/test_long_with_subgroups/'
        state_size = 2

        c_train = deepcopy(c_train_template)
        c_train.io['data_files'] = [os.path.join(train_dir, file) for file in train_files]
        c_train.io['num_evo_entries'] = 42
        c_train.architecture['recurrent_layer_size'] = [state_size]
        c_train.optimization['batch_size'] = 224
        c_train.optimization['num_steps'] = 700

        c_eval = deepcopy(c_train)
        c_eval.io['data_files'] = [os.path.join(eval_dir, file) for file in eval_files]
        c_eval.io['evaluation_sub_groups'] = ['10', '20', '30', '40', '50', '70', '90']
        c_eval.optimization['min_after_dequeue'] = 1

        npr.seed(1)
        w = {'rnn/lstm_cell/kernel':    (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'rnn/lstm_cell/bias':     (npr.rand(state_size * 4) - 0.5) * 0.15,
             'rnn/lstm_cell/w_f_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_i_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_o_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'linear_dihedrals/weights': (npr.rand(state_size, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':  ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

        self._testCore(c_train, [c_eval], w, {'all/loss': [[3969.7769], [18472.499], [26118.362]],
                                              '10/loss':  [[2582.1846], [11427.982], [16384.59]],
                                              '20/loss':  [[3604.9255], [17034.642], [24168.335]],
                                              '30/loss':  [[3967.7151], [18708.87],  [26485.33]],
                                              '40/loss':  [[3692.4805], [17666.263], [25002.922]],
                                              '50/loss':  [[4572.0215], [21267.812], [29969.229]],
                                              '70/loss':  [[4958.4141], [22560.04],  [31739.334]],
                                              '90/loss':  [[4410.6983], [20641.902], [29078.809]]})

    def testEvaluationSubgroupsFirstOrderLoss(self):
        # not sourced from anywhere else. just used as a baseline for the future
        train_files = ['1', '2', '3']
        eval_files  = ['1']
        train_dir = base_dir + 'data/unofficial/tfrecord/training_long_with_subgroups/'
        eval_dir  = base_dir + 'data/unofficial/tfrecord/test_long_with_subgroups/'
        state_size = 2

        c_train = deepcopy(c_train_template)
        c_train.io['data_files'] = [os.path.join(train_dir, file) for file in train_files]
        c_train.io['num_evo_entries'] = 42
        c_train.architecture['recurrent_layer_size'] = [state_size]
        c_train.optimization['batch_size'] = 224
        c_train.optimization['num_steps'] = 700
        c_train.loss['tertiary_normalization'] = 'first'

        c_eval = deepcopy(c_train)
        c_eval.io['data_files'] = [os.path.join(eval_dir, file) for file in eval_files]
        c_eval.io['evaluation_sub_groups'] = ['10', '20', '30', '40', '50', '70', '90']
        c_eval.optimization['min_after_dequeue'] = 1

        npr.seed(1)
        w = {'rnn/lstm_cell/kernel':    (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'rnn/lstm_cell/bias':     (npr.rand(state_size * 4) - 0.5) * 0.15,
             'rnn/lstm_cell/w_f_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_i_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_o_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'linear_dihedrals/weights': (npr.rand(state_size, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':  ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

        self._testCore(c_train, [c_eval], w, {'all/loss': [[6072.3743], [37606.458], [27076.132]],
                                              '10/loss':  [[4488.7554], [28267.255], [19971.959]],
                                              '20/loss':  [[5189.6362], [33110.999], [23640.175]],
                                              '30/loss':  [[5623.7797], [35508.456], [25480.51]],
                                              '40/loss':  [[5063.7138], [32391.776], [23141.081]],
                                              '50/loss':  [[6492.1356], [39758.215], [28776.514]],
                                              '70/loss':  [[7477.3926], [45244.659], [32810.721]],
                                              '90/loss':  [[7095.7344], [42928.339], [31128.69]]})

    def testTwoFragmentReconstruction(self):
        c_train = deepcopy(c_train_template)
        c_train.computing['num_reconstruction_fragments'] = 2

        c_eval_wt_val = deepcopy(c_eval_template)
        c_eval_wt_val.computing['num_reconstruction_fragments'] = 2

        self._testCore(c_train, [c_eval_wt_val], w_template,
                       {'all/loss': [[1117.3124301441253], [1528.6822316674461], [8918.7761559069913]],
                        'drmsds': [[[1259.07786793,   803.25406635,  1205.25539413,  1025.61087985, 1676.55300594,   1308.98564775,   1153.30808244,   854.51602809,  815.66860265,  1004.5901707 ,   1196.04068161,   1451.05497598,  1286.43255861,   1176.03539134,    589.94585875,   1106.00902386,  1658.48477058,   1238.3419573 ,   1041.211955  ,    908.69045666,  696.5834041 ,    844.03238061,   1333.24227589,    919.05794194, 1259.16899425,   1238.97081143]],
                                   [[1852.03230968,  1298.21547954,  1701.79515487,  1546.85502048, 2432.2781724 ,   1926.46521634,   1672.05336474,  1102.57260516,  990.31405429,  1403.04726402,   1860.18546927,   2213.50226333,  1885.86486933,   1644.2164523 ,    639.60193536,   1603.07148149,  2406.45469603,   1746.76373983,   1470.15775544,   1242.50577499,  811.55817662,   1237.96052076,   1937.86330937,   1344.84908538,  849.91827047,    925.63558189]],
                                   [[10398.96744671, 9410.69248146,  9624.58117421,  9229.32326745, 12797.198093 ,  11281.55467097,  10034.56338289,   6177.7895715, 5779.85300392,  7942.10719805,  11387.75744491,  12228.81485384, 10716.13548079,   9030.89986508,   4245.08641234,   9650.40891751, 12667.68625944,   9579.74124615,   8347.40016602,   7528.6852053 , 5495.06240634,   8121.44738695,  10675.87702396,   8034.47102478, 8552.71371788,   2949.36235213]]]})

    def testFourFragmentReconstruction(self):
        c_train = deepcopy(c_train_template)
        c_train.computing['num_reconstruction_fragments'] = 4

        c_eval_wt_val = deepcopy(c_eval_template)
        c_eval_wt_val.computing['num_reconstruction_fragments'] = 4

        self._testCore(c_train, [c_eval_wt_val], w_template,
                       {'all/loss': [[1117.3124301441253], [1528.6822316674461], [8918.7761559069913]],
                        'drmsds': [[[1259.07786793,   803.25406635,  1205.25539413,  1025.61087985, 1676.55300594,   1308.98564775,   1153.30808244,   854.51602809,  815.66860265,  1004.5901707 ,   1196.04068161,   1451.05497598,  1286.43255861,   1176.03539134,    589.94585875,   1106.00902386,  1658.48477058,   1238.3419573 ,   1041.211955  ,    908.69045666,  696.5834041 ,    844.03238061,   1333.24227589,    919.05794194, 1259.16899425,   1238.97081143]],
                                   [[1852.03230968,  1298.21547954,  1701.79515487,  1546.85502048, 2432.2781724 ,   1926.46521634,   1672.05336474,  1102.57260516,  990.31405429,  1403.04726402,   1860.18546927,   2213.50226333,  1885.86486933,   1644.2164523 ,    639.60193536,   1603.07148149,  2406.45469603,   1746.76373983,   1470.15775544,   1242.50577499,  811.55817662,   1237.96052076,   1937.86330937,   1344.84908538,  849.91827047,    925.63558189]],
                                    [[10398.96744671, 9410.69248146,  9624.58117421,  9229.32326745, 12797.198093 ,  11281.55467097,  10034.56338289,   6177.7895715, 5779.85300392,  7942.10719805,  11387.75744491,  12228.81485384, 10716.13548079,   9030.89986508,   4245.08641234,   9650.40891751, 12667.68625944,   9579.74124615,   8347.40016602,   7528.6852053 , 5495.06240634,   8121.44738695,  10675.87702396,   8034.47102478, 8552.71371788,   2949.36235213]]]})

    def testOneFragmentReconstruction(self):
        c_train = deepcopy(c_train_template)
        c_train.computing['num_reconstruction_fragments'] = 1

        c_eval_wt_val = deepcopy(c_eval_template)
        c_eval_wt_val.computing['num_reconstruction_fragments'] = 1

        self._testCore(c_train, [c_eval_wt_val], w_template,
                       {'all/loss': [[1117.3124301441253], [1528.6822316674461], [8918.7761559069913]],
                        'drmsds': [[[1259.07786793,   803.25406635,  1205.25539413,  1025.61087985, 1676.55300594,   1308.98564775,   1153.30808244,   854.51602809,  815.66860265,  1004.5901707 ,   1196.04068161,   1451.05497598,  1286.43255861,   1176.03539134,    589.94585875,   1106.00902386,  1658.48477058,   1238.3419573 ,   1041.211955  ,    908.69045666,  696.5834041 ,    844.03238061,   1333.24227589,    919.05794194, 1259.16899425,   1238.97081143]],
                                   [[1852.03230968,  1298.21547954,  1701.79515487,  1546.85502048, 2432.2781724 ,   1926.46521634,   1672.05336474,  1102.57260516,  990.31405429,  1403.04726402,   1860.18546927,   2213.50226333,  1885.86486933,   1644.2164523 ,    639.60193536,   1603.07148149,  2406.45469603,   1746.76373983,   1470.15775544,   1242.50577499,  811.55817662,   1237.96052076,   1937.86330937,   1344.84908538,  849.91827047,    925.63558189]],
                                    [[10398.96744671, 9410.69248146,  9624.58117421,  9229.32326745, 12797.198093 ,  11281.55467097,  10034.56338289,   6177.7895715, 5779.85300392,  7942.10719805,  11387.75744491,  12228.81485384, 10716.13548079,   9030.89986508,   4245.08641234,   9650.40891751, 12667.68625944,   9579.74124615,   8347.40016602,   7528.6852053 , 5495.06240634,   8121.44738695,  10675.87702396,   8034.47102478, 8552.71371788,   2949.36235213]]]})

    def testConstantLengthCurriculum(self):
        train_dir = base_dir + 'data/CASP11Thinning30TwoResidueShiftEvoUniParcBakerJackHMMERNeg10JackHMMERNeg10/training/full/'
        eval_dir = base_dir + 'data/CASP11Thinning30TwoResidueShiftEvoUniParcBakerJackHMMERNeg10JackHMMERNeg10/validation/sample/'
        train_files = ['1', '2', '3']
        eval_files = ['1']

        state_size = 2
        batch_size = 224
        max_seq_length = 700

        npr.seed(1)
        w = {'rnn/lstm_cell/kernel':     (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'rnn/lstm_cell/bias':       (npr.rand(state_size * 4) - 0.5) * 0.15,
             'rnn/lstm_cell/w_f_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_i_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_o_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'linear_dihedrals/weights': (npr.rand(state_size, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':  ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

        c_train = deepcopy(c_train_template)
        c_train.io['data_files'] = [os.path.join(train_dir, file) for file in train_files]
        c_train.io['num_evo_entries'] = 42
        c_train.architecture['recurrent_layer_size'] = [state_size]
        c_train.optimization['batch_size'] = batch_size
        c_train.optimization['num_steps'] = max_seq_length
        c_train.loss['tertiary_normalization'] = 'first'
        c_train.curriculum['mode'] = 'length'
        c_train.curriculum['behavior'] = 'constant'
        c_train.curriculum['base'] = 100.0

        c_eval_wt_val = deepcopy(c_train)
        c_eval_wt_val.io['data_files'] = [os.path.join(eval_dir, file) for file in eval_files]

        c_eval_unwt_val = deepcopy(c_eval_wt_val)
        c_eval_unwt_val.curriculum['mode'] = None
        c_eval_unwt_val.curriculum['behavior'] = None

        self._testCore(c_train, [c_eval_wt_val, c_eval_unwt_val], w,
                       {'all/loss': [[1172.3326, 6072.3743], [3056.477, 19040.947], [7080.9662, 38034.351]]})

    def testLossChangeLengthCurriculumAndHistoryUpdating(self):
        train_dir = base_dir + 'data/CASP11Thinning30TwoResidueShiftEvoUniParcBakerJackHMMERNeg10JackHMMERNeg10/training/full/'
        eval_dir = base_dir + 'data/CASP11Thinning30TwoResidueShiftEvoUniParcBakerJackHMMERNeg10JackHMMERNeg10/validation/sample/'
        train_files = ['1', '2', '3']
        eval_files = ['1']

        state_size = 2
        batch_size = 224
        max_seq_length = 700

        npr.seed(1)
        w = {'rnn/lstm_cell/kernel':     (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'rnn/lstm_cell/bias':       (npr.rand(state_size * 4) - 0.5) * 0.15,
             'rnn/lstm_cell/w_f_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_i_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_o_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'linear_dihedrals/weights': (npr.rand(state_size, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':  ((npr.rand(output_size) - 0.5) * 0.05) + 0.2,
             'curriculum_loss_history':  np.array([-1., -1, 10., 10., 10.])}

        c_train = deepcopy(c_train_template)
        c_train.io['data_files'] = [os.path.join(train_dir, file) for file in train_files]
        c_train.io['num_evo_entries'] = 42
        c_train.architecture['recurrent_layer_size'] = [state_size]
        c_train.queueing['min_after_dequeue'] = 1
        c_train.queueing['batch_queue_capacity'] = 1
        c_train.optimization['batch_size'] = batch_size
        c_train.optimization['num_steps'] = max_seq_length
        c_train.optimization['learning_rate'] = 0.0001
        c_train.loss['tertiary_normalization'] = 'first'
        c_train.curriculum['mode'] = 'length'
        c_train.curriculum['behavior'] = 'loss_change'
        c_train.curriculum['base'] = 100.0
        c_train.curriculum['rate'] = 50.0
        c_train.curriculum['sharpness'] = 0.1
        c_train.curriculum['slope'] = 1.0
        c_train.curriculum['change_num_iterations'] = 5

        c_eval_unwt_val = deepcopy(c_train)
        c_eval_unwt_val.io['data_files'] = [os.path.join(eval_dir, file) for file in eval_files]
        c_eval_unwt_val.curriculum['mode'] = None
        c_eval_unwt_val.curriculum['behavior'] = None
        c_eval_unwt_val.curriculum['update_loss_history'] = True

        self._testCore(c_train, [c_eval_unwt_val], w,
                       {'all/loss': [[6072.3743], [3292.0307], [6780.2803], [1738.969], [13086.583]]},
                       {'curriculum_step': [[100.0], [100.0], [134.60767], [168.0144], [207.8342]],
                        'curriculum_loss_history': [[[-1., -1., 10., 10., 10.]], [[-1., 10., 10., 10., 60.723743]], [[10., 10., 10., 60.723743, 32.920307]], [[10., 10., 60.723743, 32.920307, 67.802803]], [[10., 60.723743, 32.920307, 67.802803, 17.38969]]]})

    def testLossChangeLengthCurriculumAndHistoryUpdating(self):
        c_train = deepcopy(c_train_template)
        c_train.curriculum['mode'] = 'loss'
        c_train.curriculum['behavior'] = 'loss_change'
        c_train.curriculum['slope'] = 1.0
        c_train.curriculum['base'] = 40.0
        c_train.curriculum['rate'] = 10.0
        c_train.curriculum['change_num_iterations'] = 5
        c_train.curriculum['sharpness'] = 0.1

        c_eval_wt_val = deepcopy(c_train)
        c_eval_wt_val.io['data_files']                  = c_eval_template.io['data_files']
        c_eval_wt_val.optimization['batch_size']        = c_eval_template.optimization['batch_size']
        c_eval_wt_val.curriculum['update_loss_history'] = True

        npr.seed(1)
        w = {'rnn/lstm_cell/kernel':    (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'rnn/lstm_cell/bias':     (npr.rand(state_size * 4) - 0.5) * 0.15,
             'rnn/lstm_cell/w_f_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_i_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_o_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'linear_dihedrals/weights': (npr.rand(state_size, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':  ((npr.rand(output_size) - 0.5) * 0.05) + 0.2,
             'curriculum_loss_history':  np.array([-1., -1, 10., 10., 10.])}

        # values sourced from autograd, tests for 5 rather than the usual 2 steps. Need to lower tolerances because at the last step numerical differences really add up.
        # The semantics of this section are not necessarily how the TF-based scheme will ultimately work, 
        # but I just want something that's equivalent to the autograd version so that I can check for bugs.
        self._testCore(c_train, [c_eval_wt_val], w,
                       {'all/loss': [[811.62446811817381], [2856.1590688980946], [4344.9414086697989], [2582.6861198060337], [6981.8546053318314], [1736.7348480158662]],
                        'drmsds': [[[712.40290179,   656.46970474,   790.84479972,   652.82797677,  961.66260809,   767.57335483,   874.35258872,   806.63533543,  782.99623018,   749.18344109,   738.81951414,   774.31806268,  827.08632313,   867.14142333,   589.94743712,   820.71294084,  942.09196521,   918.12720528,   813.31124777,   822.39872622,  698.84897879,   722.90436389,   770.49497536,   649.19058281,  1152.92809197,  1238.96539117]],
                                   [[3072.38719134,  3008.42793991,  2911.85857315,  3065.61335431, 3168.97061669,  2941.77352842,  3017.70879495,  2891.57130501,  2701.93006486,  3019.22871584,  3068.32486473,  3267.78362481,  3045.9070289 ,  3093.56330986,  2006.90778867,  3057.33263582,  3186.7607088 ,  3166.60588134,  3106.6484276 ,  3006.92982362,  2557.87729174,  2975.23974889,  3033.74464829,  2996.44252548,  2141.96776781,   748.62963052]],
                                   [[4781.97681807,  4610.20183203,  4625.65332841,  4681.39396872, 4943.34334964,  4645.67152515,  4773.59273225,  3960.64553748,  3666.32370734,  4547.68215512,  4817.98980224,  5047.5953698 ,  4803.93737397,  4733.75131504,  2623.49181102,  4714.21210284,  4970.23894949,  4784.20792262,  4662.23881565,  4426.60728458,  3422.24637855,  4520.60771824,  4770.66415055,  4526.47017802,  3553.4400979 ,  1354.29240071]],
                                   [[2967.80588702,  2734.33762731,  2902.14601824,  2851.89942391, 3131.98661834,  2820.8206015 ,  2971.57460481,  2179.09722376,  1959.44547603,  2757.53174217,  2975.04859064,  3209.30119375,  3053.09449444,  2927.63784721,  1312.8612172 ,  2865.69773776,  3197.78200022,  2980.01981997,  2870.4279071 ,  2573.26712122,  1742.72926226,  2649.08371411,  3019.5858932 ,  2709.04971701,  1583.53807956,   204.06929621]],
                                   [[8031.97246028,  7560.39327618,  7765.67768081,  7607.87563196, 8553.09061973,  8036.12360195,  7888.60174855,  5300.36816215,  4937.32372891,  6823.85976053,  8202.36916539,  8542.55818638,  8139.08316523,  7552.77427996,  3598.95793179,  7713.18990218,  8574.9111079 ,  7783.99178953,  7155.05125404,  6448.80506888,  4667.93060252,  6927.8142766 ,  8121.28587583,  6894.19364023,  6381.82601701,  2318.19080412]],
                                   [[2138.72860162,  1634.22427187,  2027.21185137,  1895.23934007, 2390.65907149,  2008.27670831,  1991.1240206 ,  1342.20067604,  1186.5824904 ,  1712.56864805,  2053.95511362,  2307.17673441,  2120.20372179,  1949.00693428,   784.49138156,  1932.92812112,  2337.1205167 ,  2055.09641836,  1808.83662235,  1508.9627241 ,   991.81131387,  1598.10387413,  2173.9986364 ,  1685.60747253,   813.74856755,   707.24221584]]]},
                       {'curriculum_step': [[40.0], [40.0], [47.62279905095761], [54.627107638682787], [61.798503765010253], [68.57091859250194]],
                        'curriculum_loss_history': [[[-1., -1., 10., 10., 10.]], [[-1., 10., 10., 10., 8.11624146]], [[10.0, 10.0, 10.0, 8.1162447068527701, 28.561590903221521]], [[10.0, 10.0, 8.1162447068527701, 28.561590903221521, 43.449414498621017]], [[10.0, 8.1162447068527701, 28.561590903221521, 43.449414498621017, 25.826861006832079]], [[8.1162447068527701, 28.561590903221521, 43.449414498621017, 25.826861006832079, 69.818545616970383]]]},
                       rtol=1e-3, atol=1)

    def testEvaluationSubgroupsFirstOrderBatchIndependentLoss(self):
        # not sourced from anywhere else. just used as a baseline for the future, but new denominators were manually checked to insure correct semantics.
        train_files = ['1', '2', '3']
        eval_files  = ['1']
        train_dir = base_dir + 'data/unofficial/tfrecord/training_long_with_subgroups/'
        eval_dir  = base_dir + 'data/unofficial/tfrecord/test_long_with_subgroups/'
        state_size = 2

        c_train = deepcopy(c_train_template)
        c_train.io['data_files'] = [os.path.join(train_dir, file) for file in train_files]
        c_train.io['num_evo_entries'] = 42
        c_train.architecture['recurrent_layer_size'] = [state_size]
        c_train.optimization['batch_size'] = 224
        c_train.optimization['num_steps'] = 700
        c_train.loss['tertiary_normalization'] = 'first'
        c_train.loss['batch_dependent_normalization'] = False

        c_eval = deepcopy(c_train)
        c_eval.io['data_files'] = [os.path.join(eval_dir, file) for file in eval_files]
        c_eval.io['evaluation_sub_groups'] = ['10', '20', '30', '40', '50', '70', '90']
        c_eval.optimization['min_after_dequeue'] = 1

        npr.seed(1)
        w = {'rnn/lstm_cell/kernel':    (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'rnn/lstm_cell/bias':     (npr.rand(state_size * 4) - 0.5) * 0.15,
             'rnn/lstm_cell/w_f_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_i_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_o_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'linear_dihedrals/weights': (npr.rand(state_size, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':  ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

        self._testCore(c_train, [c_eval], w, {'all/loss': [[1791.0208], [12798.95], [12497.742]],
                                              '10/loss':  [[867.97943], [6317.3691], [6167.0677]],
                                              '20/loss':  [[1400.5097], [10328.725], [10083.11]],
                                              '30/loss':  [[1712.145 ], [12480.058], [12185.418]],
                                              '40/loss':  [[1460.2353], [10788.871], [10533.414]],
                                              '50/loss':  [[2101.8034], [14844.029], [14495.937]],
                                              '70/loss':  [[2649.4751], [18480.762], [18048.781]],
                                              '90/loss':  [[2344.9177], [16352.843], [15970.471]]})

    def testEvaluationSubgroupsSecondOrderBatchIndependentLoss(self):
        # not sourced from anywhere else. just used as a baseline for the future, but new denominators were manually checked to insure correct semantics.
        train_files = ['1', '2', '3']
        eval_files  = ['1']
        train_dir = base_dir + 'data/unofficial/tfrecord/training_long_with_subgroups/'
        eval_dir  = base_dir + 'data/unofficial/tfrecord/test_long_with_subgroups/'
        state_size = 2

        c_train = deepcopy(c_train_template)
        c_train.io['data_files'] = [os.path.join(train_dir, file) for file in train_files]
        c_train.io['num_evo_entries'] = 42
        c_train.architecture['recurrent_layer_size'] = [state_size]
        c_train.optimization['batch_size'] = 224
        c_train.optimization['num_steps'] = 700
        c_train.loss['tertiary_normalization'] = 'second'
        c_train.loss['batch_dependent_normalization'] = False

        c_eval = deepcopy(c_train)
        c_eval.io['data_files'] = [os.path.join(eval_dir, file) for file in eval_files]
        c_eval.io['evaluation_sub_groups'] = ['10', '20', '30', '40', '50', '70', '90']
        c_eval.optimization['min_after_dequeue'] = 1

        npr.seed(1)
        w = {'rnn/lstm_cell/kernel':    (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
             'rnn/lstm_cell/bias':     (npr.rand(state_size * 4) - 0.5) * 0.15,
             'rnn/lstm_cell/w_f_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_i_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'rnn/lstm_cell/w_o_diag':   (npr.rand(state_size) - 0.5) * 0.05,
             'linear_dihedrals/weights': (npr.rand(state_size, output_size) - 0.5) * 0.05,
             'linear_dihedrals/biases':  ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

        self._testCore(c_train, [c_eval], w, {'all/loss': [[1013.3944], [5759.3643], [4065.358]],
                                              '10/loss':  [[415.19384], [2400.3902], [1690.2597]],
                                              '20/loss':  [[668.11881], [3933.5342], [2760.5053]],
                                              '30/loss':  [[870.12339], [5066.3235], [3568.9159]],
                                              '40/loss':  [[696.99821], [4124.3553], [2900.0895]],
                                              '50/loss':  [[1184.4842], [6638.0241], [4693.0023]],
                                              '70/loss':  [[1703.0519], [9556.1241], [6757.6286]],
                                              '90/loss':  [[1555.7903], [8596.7957], [6087.1048]]})


class IdiosyncraticTest(tf.test.TestCase):
    """ Mishmash of idiosyncratic tests. """

    def setUp(self):
        super(IdiosyncraticTest, self).setUp()
        tf.logging.error("Starting: %s", self._testMethodName)

    def tearDown(self):
        super(IdiosyncraticTest, self).tearDown()
        tf.logging.error("Finished: %s", self._testMethodName)

    def testEvaluation(self):
        with self.test_session(use_gpu=use_gpu) as sess:
            m_train = RGNModel('training', c_train_template)
            m_eval = RGNModel('evaluation', c_eval_template)

            m_train.start([m_eval], sess, False)
            assign_weights(sess, w_template)

            try:
                for l_expected in [11.173124301441253, 15.286822316674461, 89.187761559069913]:
                    l_actual = m_eval.evaluate(sess)['tertiary_loss_all']
                    self.assertAllClose(l_expected, l_actual, rtol=5e-2, atol=5e-2)
                    m_train.train(sess)
            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False)  

    def testTrainingLoss(self):
        with self.test_session(use_gpu=use_gpu) as sess:
            m_train = RGNModel('training', c_train_template)

            m_train.start([], sess, False)
            assign_weights(sess, w_template)

            d_actual, l_actual = sess.run(get_node_ops(['model_0/drmsds', 'model_0/all/loss']))
            
            l_expected = 938.64895491646837
            d_expected = [1199.6802679570121, 959.05839594905524, 896.68441273265944, 1019.6666910685933, 774.17536506706131, 842.4372448966518, 1144.7650008777878, 596.73343008131951, 1211.3051877486851, 741.98355278585802]

            try:
                # values sourced from autograd
                self.assertAllClose(l_expected, l_actual)
                self.assertAllClose(d_expected, d_actual)
            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False)

    def testPrediction(self):
        with self.test_session(use_gpu=use_gpu) as sess:
            m_train = RGNModel('training', c_train_template)
            m_eval = RGNModel('evaluation', c_eval_template)

            m_train.start([m_eval], sess, False)
            assign_weights(sess, w_template)

            try:
                # expected predictions were originally sourced from autograd but eventually they had to be sourced from TF after I changed
                # the initialization coordinates in points_to_coordinates. however, care was taken to insure that no errors were introduced 
                # due to the shift to new initialization coordinates. the values are stored on disk because they're too much to dump here.
                for i in range(1, 3):
                    m_train.train(sess)
                    preds_actual = {id_: dict_['tertiary'] for id_, dict_ in m_eval.predict(sess).iteritems()}
                    with open(artifacts_dir + 'predictions' + str(i), 'r') as f: preds_expected = literal_eval(f.read())
                    for pred_actual, pred_expected in dicts_to_matched_tuples(preds_actual, preds_expected):
                       self.assertAllClose(pred_expected, pred_actual, rtol=1e-1, atol=1e-1)
            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False)

    def testDiagnosticTracking(self):
        with self.test_session(use_gpu=use_gpu) as sess:
            c_train = deepcopy(c_train_template)
            c_train.curriculum['mode'] = 'loss'
            c_train.curriculum['behavior'] = 'loss_threshold'
            c_train.curriculum['slope'] = 1.0
            c_train.curriculum['base'] = 40.0
            c_train.curriculum['rate'] = 10.0
            c_train.curriculum['threshold'] = 20.0 * 100 # losses are internally not in angstroms
            c_train.curriculum['change_num_iterations'] = 3 # don't need this here but want to lower it to test loss history pruning functionality

            c_eval_wt_val = deepcopy(c_train)
            c_eval_wt_val.io['data_files']                  = c_eval_template.io['data_files']
            c_eval_wt_val.optimization['batch_size']        = c_eval_template.optimization['batch_size']
            c_eval_wt_val.loss['update_loss_history'] = True

            m_train = RGNModel('training', c_train)
            m_eval_wt_val = RGNModel('evaluation', c_eval_wt_val)

            m_train.start([m_eval_wt_val], sess, False)
            assign_weights(sess, w_template)

            ds_expected = [{'curriculum_quantiles': [11, 21, 31, 42], 'curriculum_step': 40.0, 'max_grad': 22.979836, 'max_weight': 0.212669,    'min_grad': -475.43021, 'min_weight': -0.074623868},
                           {'curriculum_quantiles': [13, 26, 38, 51], 'curriculum_step': 50.0, 'max_grad': 343.01083, 'max_weight': 0.2078426,   'min_grad': -17.459183, 'min_weight': -0.074623466},
                           {'curriculum_quantiles': [16, 31, 45, 61], 'curriculum_step': 60.0, 'max_grad': 277.08871, 'max_weight': 0.084469058, 'min_grad': -1966.3207, 'min_weight': -0.46561706}]

            try:
                for d_expected in ds_expected:
                    m_eval_wt_val.evaluate(sess) # this is needed to update loss history
                    d_actual = m_train.diagnose(sess) # note that diagnostic evaluations consume training batches, so in total this is doing five steps.     
                    for values_actual, values_expected in dicts_to_matched_tuples(d_actual, d_expected):
                        self.assertAllClose(values_expected, values_actual, rtol=1e-2, atol=1e-2)
                    m_train.train(sess)
            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False)

    def testQueueSizes(self):
        with self.test_session(use_gpu=use_gpu) as sess:    
            m_train = RGNModel('training', c_train_template)
            m_train.start([], sess, False)

            g = tf.get_default_graph()
            
            try:
                # check file queue size
                fq_size = g.get_operation_by_name('RGN/model_0/file_queue/file_queue_Size').outputs[0]
                while fq_size.eval(session=sess) < c_train_template.queueing['file_queue_capacity']: time.sleep(20)
                self.assertAllEqual(c_train_template.queueing['file_queue_capacity'], fq_size.eval(session=sess))

                # check batching queue size
                bq_size = g.get_operation_by_name('RGN/model_0/batching_queue/padding_fifo_queue_Size').outputs[0]
                while bq_size.eval(session=sess) < c_train_template.queueing['batch_queue_capacity']: time.sleep(20)
                self.assertAllEqual(c_train_template.queueing['batch_queue_capacity'], bq_size.eval(session=sess))
            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False)             

    def testQueueShuffling(self):
        with self.test_session(use_gpu=use_gpu) as sess:
            c_train = deepcopy(c_train_template)
            c_train.queueing['shuffle'] = True
            c_train.queueing['min_after_dequeue'] = 1
            c_train.optimization['batch_size'] = 100
            c_train.io['data_files'] = [train_dir + str(i) for i in range(1, 10)]

            m_train = RGNModel('training', c_train)
            m_train.start([], sess, False)

            g = tf.get_default_graph()
            
            ids_actual = list(g.get_operation_by_name('RGN/model_0/ids').outputs[0].eval(session=sess))
            ids_unexpected = ['2', '3', '4', '7', '21', '25', '29', '38', '39', '40', '47', '50', '51', '52', '55', '56', '60', '73', '80', '89', '90', '93', '99', '102', '105', '112', '117', '119', '120', '121', '122', '123', '124', '125', '135', '144', '145', '148', '150', '154', '155', '156', '162', '172', '173', '176', '177', '178', '180', '186', '187', '188', '191', '202', '211', '214', '218', '224', '225', '226', '227', '228', '230', '231', '232', '233', '238', '240', '245', '249', '250', '251', '252', '255', '258', '265', '267', '268', '270', '271', '272', '281', '286', '287', '293', '299', '307', '309', '314', '315', '316', '321', '329', '330', '331', '332', '333', '334', '335', '336']
            
            try:
                self.assertAllEqual(False, ids_actual == ids_unexpected)
            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False) 

    def testSeedCongruenceAndRandomization(self):
        alphabet_size = 7

        c_train_1 = deepcopy(c_train_template)
        c_train_1.architecture['bidirectional'] = True
        c_train_1.architecture['tertiary_output'] = 'linear_alphabet'
        c_train_1.architecture['alphabet_size'] = alphabet_size

        c_train_2 = deepcopy(c_train_1)

        c_train_3 = deepcopy(c_train_1)
        c_train_3.initialization['graph_seed'] = 2

        d_train_1 = {}
        d_train_2 = {}
        d_train_3 = {}

        for c_train, d_train in [(c_train_1, d_train_1), (c_train_2, d_train_2), (c_train_3, d_train_3)]:
            with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
                m_train = RGNModel('training', c_train)
                m_train.start([], sess, False)

                d_train.update({var.name: var.eval(sess) for var in tf.get_collection(tf.GraphKeys.WEIGHTS)})

                m_train.finish(sess, save=False, close_session=False, reset_graph=False) 

        for k, v in d_train_1.iteritems():
            self.assertAllEqual(True,  np.array_equal(d_train_2[k], v))
            self.assertAllEqual(False, np.array_equal(d_train_3[k], v))

    def testAllPrunedTrainingSamplesConsumed(self):
        with self.test_session(use_gpu=use_gpu) as sess:
            batch_size = 128

            ids_expected = [2,3,4,7,21,25,29,38,39,40,47,50,51,52,55,56,60,73,80,89,90,93,99,102,105,112,117,119,120,121,122,123,124,125,135,144,145,148,150,154,155,156,162,172,173,176,177,178,180,186,187,188,191,202,211,214,218,224,225,226,227,228,230,231,232,233,238,240,245,249,250,251,252,255,258,265,267,268,270,271,272,281,286,287,293,299,307,309,314,315,316,321,329,330,331,332,333,334,335,336,338,340,342,345,347,348,349,352,353,357,362,369,371,379,384,386,387,392,394,396,398,401,402,405,406,409,411,413,418,419,423,424,425,426,427,428,437,438,441,443,448,449,450,451,453,460,461,462,463,464,468,471,482,483,491,496,500,502,504,515,519,521,522,523,524,531,535,537,538,539,540,543,544,546,547,550,555,559,561,562,565,577,582,583,585,587,590,592,602,603,604,605,608,615,621,628,629,631,636,637,640,642,643,646,648,649,650,655,656,657,659,667,670,680,681,683,684,687,690,692,695,706,707,709,712,713,714,715,717,719,720,722,723,724,725,726,727,729,739,740,741,743,744,745,746,748,749,752,753,755,756,758,760,761,762,764,765,769,771,772,773,774,775,778,779,784,786,787,788,789,790,791,792,793,794,795,796,797,800,802,803,805,806,810,811,812,816,817,820,824,825,827,831,837,838,839,841,848,849,851,852,860,861,864,869,870,877,879,880,881,882,891,894,898,899,901,903,904,910,912,914,917,921,922,923,926,927,928,929,930,931,932,933,934,935,937,939,942,943,944,949,954,960,963,975,983,984,991,992,1001,1002,1003,1005,1006,1007,1008,1009,1010,1011,1014,1015,1018,1020,1022,1023,1024,1025,1027,1028,1029,1030,1034,1035,1036,1037,1039,1041,1046,1047,1051,1052,1054,1057,1058,1059,1060,1061,1063,1065,1068,1071,1076,1078,1080,1081,1082,1083,1085,1086,1090,1091,1093,1096,1100,1105,1106,1107,1110,1111,1114,1118,1123,1124,1125,1130,1133,1135,1136,1137,1138,1144,1145,1148,1150,1151,1154,1155,1156,1160,1162,1168,1169,1171,1179,1181,1182,1183,1187,1191,1195,1196,1197,1198,1210,1213,1214,1215,1218,1225,1228,1230,1234,1237,1238,1243,1246,1247,1250,1252,1256,1260,1261,1263,1269,1270,1279,1280,1284,1291,1292,1293,1299,1306,1307,1310,1311,1315,1317,1318,1326,1332,1333,1343,1346,1352,1355,1357,1359,1360,1362,1363,1365,1371,1373,1380,1384,1386,1387,1396,1397,1399,1405,1406,1408,1409,1411,1415,1417,1422,1427,1432,1436,1443,1445,1447,1448,1451,1454,1456,1457,1459,1461,1468,1473,1479,1480,1481,1482,1483,1484,1485,1492,1496,1498,1499,1501,1504,1505,1507,1509,1510,1513,1522,1527,1528,1529,1530,1536,1539,1547,1548,1549,1554,1558,1562,1563,1565,1568,1572,1577,1580,1581,1586,1587,1588,1589,1590,1591,1594,1597,1599,1602,1605,1611,1616,1617,1619,1624,1626,1627,1629,1631,1641,1642,1643,1645,1651,1667,1668,1670,1673,1676,1681,1689,1691,1692,1693,1694,1695,1702,1707,1708,1709,1711,1712,1713,1718,1728,1733,1737,1740,1742,1743,1744,1746,1753,1764,1765,1766,1775,1777,1795,1797,1799,1801,1802,1803,1805,1807,1808,1810,1811,1813,1814,1819,1820,1821,1829,1833,1834,1836,1837,1838,1841,1842,1844,1846,1847,1857,1861,1863,1864,1866,1876,1878,1882,1884,1885,1887,1888,1895,1904,1905,1908,1910,1915,1916,1919,1920,1926,1928,1931,1933,1946,1949,1953,1956,1957,1958,1959,1960,1961,1963,1970,1976,1980,1981,1988,1991,1995,1997,2004,2005,2006,2008,2030,2031,2032,2034,2035,2036,2037,2040,2041,2051,2052,2054,2059,2061,2062,2064,2071,2072,2073,2074,2075,2079,2080,2081,2084,2085,2086,2090,2092,2100,2101,2103,2104,2105,2111,2112,2114,2119,2120,2121,2122,2124,2134,2138,2139,2140,2142,2149,2150,2160,2161,2164,2166,2168,2171,2174,2175,2177,2185,2188,2192,2196,2204,2208,2209,2212,2213,2220,2225,2235,2237,2238,2241,2244,2248,2258,2263,2264,2273,2284,2285,2288,2293,2294,2295,2298,2301,2302,2303,2304,2308,2309,2310,2311,2313,2314,2322,2329,2332,2333,2337,2338,2341,2346,2352,2354,2357,2358,2369,2372,2373,2384,2387,2389,2393,2396,2399,2401,2404,2406,2409,2413,2414,2415,2420,2421,2422,2423,2427,2434,2435,2436,2438,2441,2449,2450,2451,2452,2453,2456,2461,2462,2463,2465,2473,2474,2475,2476,2482,2483,2486,2496,2497,2502,2503,2506,2508,2511,2513,2516,2518,2520,2522,2525,2528,2529,2531,2532,2533,2536,2538,2547,2551,2552,2558,2561,2562,2563,2566,2567,2569,2576,2583,2587,2589,2591,2592,2595,2596,2608,2613,2614,2623,2626,2635,2639,2642,2643,2645,2660,2665,2667,2670,2671,2672,2673,2674,2675,2676,2677,2678,2679,2680,2682,2683,2687,2688,2689,2694,2698,2707,2713,2721,2722,2726,2733,2739,2746,2751,2754,2756,2757,2758,2766,2770,2774,2776,2777,2783,2784,2786,2787,2793,2799,2804,2805,2810,2812,2813,2817,2821,2823,2826,2831,2832,2840,2844,2849,2851,2853,2861,2863,2864,2865,2879,2882,2884,2890,2892,2896,2897,2899,2902,2904,2909,2910,2914,2915,2917,2920,2924,2926,2929,2934,2935,2939,2943,2945,2946,2947,2959,2965,2967,2970,2973,2974,2976,2983,2989,2990,2991,2994,3000,3004,3013,3019,3020,3021,3023,3024,3028,3030,3031,3032,3037,3039,3041,3043,3046,3050,3051,3052,3056,3059,3060,3064,3066,3067,3072,3075,3079,3089,3091,3094,3098,3099,3102,3103,3106,3109,3119,3120,3124,3125,3129,3133,3136,3137,3138,3139,3140,3141,3142,3145,3147,3152,3155,3156,3157,3159,3164,3173,3177,3178,3182,3186,3188,3192,3193,3194,3198,3199,3200,3201,3203,3214,3220,3223,3229,3240,3245,3252,3258,3259,3268,3270,3272,3273,3276,3279,3282,3283,3287,3291,3292,3293,3297,3298,3301,3302,3303,3305,3306,3310,3311,3312,3313,3321,3324,3326,3327,3328,3329,3332,3333,3334,3338,3344,3345,3349,3353,3357,3366,3371,3377,3380,3384,3388,3390,3391,3400,3409,3410,3411,3413,3426,3433,3438,3447,3448,3449,3450,3451,3456,3457,3458,3463,3468,3470,3474,3481,3483,3490,3493,3497,3505,3506,3513,3514,3519,3520,3521,3525,3542,3543,3547,3551,3552,3553,3555,3556,3557,3559,3560,3561,3562,3565,3569,3570,3572,3573,3583,3586,3590,3593,3595,3596,3597,3599,3600,3603,3609,3611,3617,3622,3627,3631,3632,3633,3637,3638,3645,3651,3658,3661,3667,3670,3671,3684,3688,3691,3696,3698,3701,3706,3711,3713,3718,3719,3722,3725,3728,3731,3733,3743,3747,3750,3752,3754,3755,3760,3764,3773,3774,3776,3777,3781,3787,3789,3790,3791,3792,3793,3800,3801,3806,3815,3820,3831,3835,3836,3837,3839,3842,3847,3853,3859,3860,3861,3863,3864,3875,3876,3877,3878,3879,3880,3882,3886,3890,3894,3895,3896,3898,3900,3902,3903,3904,3907,3908,3916,3924,3928,3932,3933,3934,3942,3944,3945,3951,3960,3964,3970,3972,3975,3976,3977,3983,3986,3996,3997,3999,4002,4003,4006,4008,4009,4016,4018,4019,4020,4021,4028,4030,4035,4047,4050,4059,4060,4061,4066,4067,4069,4070,4072,4074,4075,4078,4079,4082,4085,4089,4090,4093,4099,4110,4111,4114,4116,4117,4126,4130,4132,4133,4145,4147,4148,4156,4162,4163,4166,4167,4168,4171,4173,4174,4176,4179,4180,4183,4191,4192,4196,4198,4203,4205,4206,4207,4213,4214,4216,4220,4226,4228,4231,4249,4256,4259,4261,4262,4264,4265,4266,4269,4270,4273,4278,4285,4286,4288,4289,4293,4294,4295,4296,4300,4303,4304,4305,4306,4311,4314,4315,4316,4318,4319,4326,4330,4332,4333,4334,4335,4343,4352,4353,4365,4366,4370,4371,4372,4377,4387,4390,4396,4405,4406,4413,4414,4417,4423,4434,4436,4439,4441,4442,4443,4444,4447,4450,4451,4456,4458,4466,4467,4471,4473,4477,4480,4488,4490,4491,4496,4497,4498,4499,4502,4503,4507,4510,4515,4525,4526,4528,4529,4533,4537,4539,4545,4547,4549,4550,4555,4562,4570,4577,4581,4592,4593,4601,4603,4604,4614,4615,4616,4619,4620,4626,4628,4630,4631,4635,4638,4639,4644,4648,4655,4656,4658,4659,4660,4668,4674,4675,4676,4680,4682,4694,4697,4698,4699,4702,4704,4708,4716,4717,4720,4721,4722,4723,4733,4736,4740,4741,4748,4753,4756,4760,4762,4766,4767,4769,4770,4771,4782,4788,4790,4794,4796,4798,4799,4805,4807,4808,4809,4814,4816,4817,4819,4831,4833,4834,4835,4836,4839,4840,4841,4843,4844,4846,4847,4849,4850,4851,4854,4856,4860,4865,4867,4875,4879,4888,4895,4896,4900,4903,4905,4907,4910,4911,4912,4915,4918,4925,4927,4928,4930,4931,4932,4937,4945,4948,4949,4952,4953,4956,4964,4969,4973,4978,4983,4984,4985,4986,4987,4989,4995,4999,5000,5002,5007,5008,5011,5012,5014,5016,5022,5023,5025,5029,5031,5033,5036,5038,5039,5044,5045,5053,5063,5064,5069,5070,5071,5073,5076,5078,5079,5080,5081,5084,5085,5089,5092,5094,5097,5107,5110,5112,5113,5118,5121,5130,5131,5138,5139,5142,5152,5154,5155,5158,5160,5163,5167,5170,5174,5175,5178,5190,5191,5196,5198,5199,5200,5204,5206,5208,5209,5212,5215,5217,5225,5226,5227,5228,5232,5235,5236,5237,5238,5240,5251,5252,5268,5275,5282,5293,5294,5296,5297,5298,5300,5305,5322,5323,5324,5326,5327,5330,5331,5334,5335,5336,5337,5343,5346,5350,5355,5356,5358,5359,5363,5366,5367,5369,5370,5372,5373,5374,5375,5377,5378,5380,5381,5383,5386,5388,5389,5390,5392,5393,5397,5399,5401,5403,5404,5405,5406,5408,5409,5411,5412,5413,5415,5420,5426,5427,5429,5430,5437,5438,5448,5450,5451,5453,5454,5457,5460,5462,5463,5468,5469,5470,5471,5473,5475,5480,5481,5483,5488,5489,5492,5496,5500,5503,5504,5505,5508,5509,5511,5512,5513,5515,5516,5518,5519,5520,5521,5522,5523,5524,5525,5526,5527,5528,5529,5530,5531,5532,5533,5539,5540,5541,5545,5547,5548,5549,5550,5551,5552,5553,5555,5556,5558,5564,5570,5571,5579,5582,5583,5585,5593,5594,5598,5601,5603,5607,5611,5612,5614,5615,5621,5626,5627,5628,5641,5642,5647,5654,5659,5660,5672,5673,5678,5688,5691,5694,5699,5703,5714,5715,5718,5719,5720,5723,5724,5728,5735,5745,5747,5753,5756,5761,5763,5764,5771,5776,5777,5778,5781,5788,5797,5800,5808,5809,5810,5812,5813,5815,5820,5823,5833,5836,5837,5838,5839,5841,5843,5848,5849,5853,5856,5863,5873,5876,5878,5880,5883,5884,5887,5903,5907,5910,5912,5916,5920,5921,5922,5924,5929,5932,5934,5937,5938,5951,5957,5960,5961,5966,5976,5977,5979,5980,5983,5986,5992,5995,5999]
            num_samples = len(ids_expected)

            c_train = deepcopy(c_train_template)
            c_train.queueing['shuffle'] = True
            c_train.queueing['min_after_dequeue'] = 1
            c_train.optimization['batch_size'] = batch_size
            c_train.io['data_files_glob'] = train_dir + '*'
            c_train.io['data_files'] = None

            m_train = RGNModel('training', c_train)
            m_train.start([], sess, False)

            g = tf.get_default_graph()

            ids_actual = set()
            for _ in range((num_samples // batch_size) + 2):
                ids_actual.update(g.get_operation_by_name('RGN/model_0/ids').outputs[0].eval(session=sess))
            ids_actual = sorted(map(int, list(ids_actual)))

            try:
                self.assertAllEqual(ids_expected, ids_actual)
            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False) 

    def testGaussianInitializers(self):
        with self.test_session(use_gpu=use_gpu) as sess:    
            c_train = deepcopy(c_train_template)        
            c_train.initialization['recurrent_init']          = {'base': {'dist': 'gaussian', 'center':  2.3, 'range': 3.1}}
            c_train.initialization['recurrent_out_proj_init'] = {'base': {'dist': 'gaussian', 'center': -4.3},               'bias': {'center':  3.0, 'range': 2.1}}

            m_train = RGNModel('training', c_train)
            m_train.start([], sess, False)

            g = tf.get_default_graph()

            try:
                d = {var.name: var.eval(session=sess).flatten() for var in tf.get_collection(tf.GraphKeys.WEIGHTS)}
                test_num = 0
                for k, v in d.iteritems():
                    if 'rnn' in k:
                        test_num = test_num + 1
                        self.assertAllClose(2.3, np.mean(v), rtol=5e-1, atol=5e-1)
                        self.assertAllClose(3.1, np.std(v),  rtol=5e-1, atol=5e-1)  
                    elif 'linear' in k:
                        test_num = test_num + 1
                        self.assertAllClose(-4.3, np.mean(v), rtol=5e-1, atol=5e-1)
                        self.assertAllClose(0.01, np.std(v),  rtol=5e-1, atol=5e-1)  

                d = {var.name: var.eval(session=sess).flatten() for var in tf.get_collection(tf.GraphKeys.BIASES)}
                for k, v in d.iteritems():
                    if 'linear' in k: # this is really quite a silly test because the sample size is only 3! 
                        test_num = test_num + 1
                        self.assertAllClose(3.0, np.mean(v), rtol=0.3, atol=1)
                        self.assertAllClose(2.1, np.std(v),  rtol=1, atol=2)  

                self.assertAllEqual(3, test_num) # to make sure all tests are visited

            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False) 

    def testUniformAndAlphabetInitializers(self):
        with self.test_session(use_gpu=use_gpu) as sess:    
            alphabet_size = 30

            c_train = deepcopy(c_train_template)  
            c_train.architecture['tertiary_output'] = 'linear_alphabet'
            c_train.architecture['alphabet_size'] = alphabet_size
            c_train.initialization['recurrent_init']          = {'base': {'dist': 'uniform', 'center': -2.3, 'range': 4.2}}
            c_train.initialization['recurrent_out_proj_init'] = {'base': {'dist': 'uniform', 'range': 2.4}, 'bias': {'dist': 'uniform', 'center': -1.6, 'range': 3.0}}
            c_train.initialization['alphabet_init']           = {'dist': 'uniform', 'range': 3.1, 'center': -10.2}
            c_train.io['data_files'] = [os.path.join(train_dir, file) for file in train_files]

            m_train = RGNModel('training', c_train)
            m_train.start([], sess, False)

            g = tf.get_default_graph()
            d = {var.name: var.eval(session=sess).flatten() for var in tf.get_collection(tf.GraphKeys.WEIGHTS)}

            try:
                test_num = 0
                for k, v in d.iteritems():
                    if 'rnn' in k and 'diag' not in k: # skipping peephole connections because there are very few weights and thus very high variance, resulting in overly sensitive tests
                        test_num = test_num + 1
                        self.assertAllClose(np.mean(v), -2.3,       rtol=5e-1, atol=5e-1)
                        self.assertAllClose(np.min(v),  -2.3 - 4.2, rtol=5e-1, atol=5e-1)
                        self.assertAllClose(np.max(v),  -2.3 + 4.2, rtol=5e-1, atol=5e-1)
                    elif 'linear_dihedrals' in k:
                        test_num = test_num + 1
                        self.assertAllClose(np.mean(v), 0,    rtol=5e-1, atol=5e-1)
                        self.assertAllClose(np.min(v),  -2.4, rtol=5e-1, atol=5e-1)
                        self.assertAllClose(np.max(v),  2.4,  rtol=5e-1, atol=5e-1)
                    elif 'alphabet' in k:
                        test_num = test_num + 1
                        self.assertAllClose(np.mean(v), -10.2,       rtol=5e-1, atol=5e-1)
                        self.assertAllClose(np.min(v),  -10.2 - 3.1, rtol=5e-1, atol=5e-1)
                        self.assertAllClose(np.max(v),  -10.2 + 3.1, rtol=5e-1, atol=5e-1)

                d = {var.name: var.eval(session=sess).flatten() for var in tf.get_collection(tf.GraphKeys.BIASES)}
                for k, v in d.iteritems():
                    if 'linear_dihedrals' in k:
                        test_num = test_num + 1
                        self.assertAllClose(np.mean(v), -1.6,     rtol=1, atol=1)
                        self.assertAllClose(np.min(v),  -1.6 - 3, rtol=1, atol=1)
                        self.assertAllClose(np.max(v),  -1.6 + 3, rtol=1, atol=1)

                self.assertAllEqual(4, test_num) # to make sure all tests are visited

            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False)

    def testTwoInvocationsZerothOrderLoss(self):
        # values sourced from single invocation runs
        with self.test_session(use_gpu=use_gpu) as sess:
            m_train = RGNModel('training', c_train_template)

            c_eval = deepcopy(c_eval_template)
            c_eval.queueing['num_evaluation_invocations'] = 2
            c_eval.optimization['batch_size'] = eval_batch_size / c_eval.queueing['num_evaluation_invocations']
            m_eval = RGNModel('evaluation', c_eval)

            m_train.start([m_eval], sess, False)
            assign_weights(sess, w_template)

            try:
                for l_expected in [11.173124301441253, 15.286822316674461, 89.187761559069913]:
                    l_actual = m_eval.evaluate(sess)['tertiary_loss_all']
                    self.assertAllClose(l_expected, l_actual, rtol=5e-2, atol=5e-2)
                    m_train.train(sess)
            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False)  

    def testThirteenInvocationsZerothOrderLoss(self):
        # values sourced from single invocation runs
        with self.test_session(use_gpu=use_gpu) as sess:
            m_train = RGNModel('training', c_train_template)

            c_eval = deepcopy(c_eval_template)
            c_eval.queueing['num_evaluation_invocations'] = 13
            c_eval.optimization['batch_size'] = eval_batch_size / c_eval.queueing['num_evaluation_invocations']
            m_eval = RGNModel('evaluation', c_eval)

            m_train.start([m_eval], sess, False)
            assign_weights(sess, w_template)

            try:
                for l_expected in [11.173124301441253, 15.286822316674461, 89.187761559069913]:
                    l_actual = m_eval.evaluate(sess)['tertiary_loss_all']
                    self.assertAllClose(l_expected, l_actual, rtol=5e-2, atol=5e-2)
                    m_train.train(sess)
            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False)  

    def testTwoInvocationsFirstOrderLoss(self):
        # values sourced from single invocation runs
        with self.test_session(use_gpu=use_gpu) as sess:
            c_train = deepcopy(c_train_template)
            c_train.loss['tertiary_normalization'] = 'first'
            m_train = RGNModel('training', c_train)

            c_eval = deepcopy(c_eval_template)
            c_eval.loss['tertiary_normalization'] = 'first'
            c_eval.queueing['num_evaluation_invocations'] = 2
            c_eval.optimization['batch_size'] = eval_batch_size / c_eval.queueing['num_evaluation_invocations']
            m_eval = RGNModel('evaluation', c_eval)

            m_train.start([m_eval], sess, False)
            assign_weights(sess, w_template)

            try:
                for l_expected in [11.653967160378857, 29.682763504342415, 94.215383972698928]:
                    l_actual = m_eval.evaluate(sess)['tertiary_loss_all']
                    self.assertAllClose(l_expected, l_actual, rtol=5e-2, atol=5e-2)
                    m_train.train(sess)
            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False)  

    def testThirteenInvocationsFirstOrderLoss(self):
        # values sourced from single invocation runs
        with self.test_session(use_gpu=use_gpu) as sess:
            c_train = deepcopy(c_train_template)
            c_train.loss['tertiary_normalization'] = 'first'
            m_train = RGNModel('training', c_train)

            c_eval = deepcopy(c_eval_template)
            c_eval.loss['tertiary_normalization'] = 'first'
            c_eval.queueing['num_evaluation_invocations'] = 13
            c_eval.optimization['batch_size'] = eval_batch_size / c_eval.queueing['num_evaluation_invocations']
            m_eval = RGNModel('evaluation', c_eval)

            m_train.start([m_eval], sess, False)
            assign_weights(sess, w_template)

            try:
                for l_expected in [11.653967160378857, 29.682763504342415, 94.215383972698928]:
                    l_actual = m_eval.evaluate(sess)['tertiary_loss_all']
                    self.assertAllClose(l_expected, l_actual, rtol=5e-2, atol=5e-2)
                    m_train.train(sess)
            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False)  

    def testTwoInvocationsSecondOrderLoss(self):
        # values sourced from single invocation runs
        with self.test_session(use_gpu=use_gpu) as sess:
            c_train = deepcopy(c_train_template)
            c_train.loss['tertiary_normalization'] = 'second'
            m_train = RGNModel('training', c_train)

            c_eval = deepcopy(c_eval_template)
            c_eval.loss['tertiary_normalization'] = 'second'
            c_eval.queueing['num_evaluation_invocations'] = 2
            c_eval.optimization['batch_size'] = eval_batch_size / c_eval.queueing['num_evaluation_invocations']
            m_eval = RGNModel('evaluation', c_eval)

            m_train.start([m_eval], sess, False)
            assign_weights(sess, w_template)

            try:
                for l_expected in [12.088778719851207, 41.728165549812493, 87.367085158190694]:
                    l_actual = m_eval.evaluate(sess)['tertiary_loss_all']
                    self.assertAllClose(l_expected, l_actual, rtol=5e-2, atol=5e-2)
                    m_train.train(sess)
            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False)  

    def testThirteenInvocationsSecondOrderLoss(self):
        # values sourced from single invocation runs
        with self.test_session(use_gpu=use_gpu) as sess:
            c_train = deepcopy(c_train_template)
            c_train.loss['tertiary_normalization'] = 'second'
            m_train = RGNModel('training', c_train)

            c_eval = deepcopy(c_eval_template)
            c_eval.loss['tertiary_normalization'] = 'second'
            c_eval.queueing['num_evaluation_invocations'] = 13
            c_eval.optimization['batch_size'] = eval_batch_size / c_eval.queueing['num_evaluation_invocations']
            m_eval = RGNModel('evaluation', c_eval)

            m_train.start([m_eval], sess, False)
            assign_weights(sess, w_template)

            try:
                for l_expected in [12.088778719851207, 41.728165549812493, 87.367085158190694]:
                    l_actual = m_eval.evaluate(sess)['tertiary_loss_all']
                    self.assertAllClose(l_expected, l_actual, rtol=5e-2, atol=5e-2)
                    m_train.train(sess)
            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False)  

    def testLossThresholdLossCurriculumAndHistoryUpdatingWithThirteenInvocations(self):
        # values sourced from single invocation runs
        with self.test_session(use_gpu=use_gpu) as sess:
            c_train = deepcopy(c_train_template)
            c_train.curriculum['mode'] = 'loss'
            c_train.curriculum['behavior'] = 'loss_threshold'
            c_train.curriculum['slope'] = 1.0
            c_train.curriculum['base'] = 40.0
            c_train.curriculum['rate'] = 10.0
            c_train.curriculum['threshold'] = 20.0
            c_train.curriculum['change_num_iterations'] = 3 # don't need this here but want to lower it to test loss history pruning functionality
            m_train = RGNModel('training', c_train)

            c_eval = deepcopy(c_train)
            c_eval.io['data_files'] = c_eval_template.io['data_files']
            c_eval.queueing['num_evaluation_invocations'] = 13
            c_eval.optimization['batch_size'] = eval_batch_size / c_eval.queueing['num_evaluation_invocations']
            c_eval.curriculum['update_loss_history'] = True
            m_eval = RGNModel('evaluation', c_eval)

            m_train.start([m_eval], sess, False)
            assign_weights(sess, w_template)

            try:
                for l_expected, lh_expected in zip([8.1162415, 35.131182, 55.600815, 9.6336652, 76.265435],
                                                   [[-1., -1., 8.1162415], [-1., 8.1162415, 35.131182], [8.1162415, 35.131182, 55.600815], [35.131182, 55.600815, 9.6336652], [55.600815, 9.6336652, 76.265435]]):
                    eval_dict = m_eval.evaluate(sess, pretty=False) 
                    l_actual = eval_dict['tertiary_loss_all']
                    lh_actual = eval_dict['update_curriculum_history_op']
                    self.assertAllClose(l_expected,  l_actual,  rtol=5e-2, atol=5e-2)
                    self.assertAllClose(lh_expected, lh_actual, rtol=5e-2, atol=5e-2)
                    m_train.train(sess)

                cs_expected = 60.0
                cs_actual = m_train.diagnose(sess)['curriculum_step']
                self.assertAllClose(cs_expected, cs_actual)
            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False)  

    def testLossChangeLossCurriculumAndHistoryUpdatingWithThirteenInvocations(self):
        # values sourced from single invocation runs
        with self.test_session(use_gpu=use_gpu) as sess:
            c_train = deepcopy(c_train_template)
            c_train.curriculum['mode'] = 'loss'
            c_train.curriculum['behavior'] = 'loss_change'
            c_train.curriculum['slope'] = 1.0
            c_train.curriculum['base'] = 40.0
            c_train.curriculum['rate'] = 10.0
            c_train.curriculum['change_num_iterations'] = 5
            c_train.curriculum['sharpness'] = 0.1
            m_train = RGNModel('training', c_train)

            c_eval = deepcopy(c_train)
            c_eval.io['data_files'] = c_eval_template.io['data_files']
            c_eval.queueing['num_evaluation_invocations'] = 13
            c_eval.optimization['batch_size'] = eval_batch_size / c_eval.queueing['num_evaluation_invocations']
            c_eval.curriculum['update_loss_history'] = True
            m_eval = RGNModel('evaluation', c_eval)

            npr.seed(1)
            w = {'rnn/lstm_cell/kernel':    (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
                 'rnn/lstm_cell/bias':     (npr.rand(state_size * 4) - 0.5) * 0.15,
                 'rnn/lstm_cell/w_f_diag':   (npr.rand(state_size) - 0.5) * 0.05,
                 'rnn/lstm_cell/w_i_diag':   (npr.rand(state_size) - 0.5) * 0.05,
                 'rnn/lstm_cell/w_o_diag':   (npr.rand(state_size) - 0.5) * 0.05,
                 'linear_dihedrals/weights': (npr.rand(state_size, output_size) - 0.5) * 0.05,
                 'linear_dihedrals/biases':  ((npr.rand(output_size) - 0.5) * 0.05) + 0.2,
                 'curriculum_loss_history':  np.array([-1., -1, 10., 10., 10.])}

            m_train.start([m_eval], sess, False)
            assign_weights(sess, w)

            try:
                for l_expected, lh_expected in zip([8.1162446811817381, 28.561590688980946, 43.449414086697989, 25.826861198060337, 69.818546053318314],
                                                   [[-1., 10., 10., 10., 8.11624146], [10.0, 10.0, 10.0, 8.1162447068527701, 28.561590903221521], [10.0, 10.0, 8.1162447068527701, 28.561590903221521, 43.449414498621017], [10.0, 8.1162447068527701, 28.561590903221521, 43.449414498621017, 25.826861006832079], [8.1162447068527701, 28.561590903221521, 43.449414498621017, 25.826861006832079, 69.818545616970383]]):
                    eval_dict = m_eval.evaluate(sess, pretty=False) 
                    l_actual = eval_dict['tertiary_loss_all']
                    lh_actual = eval_dict['update_curriculum_history_op']
                    self.assertAllClose(l_expected,  l_actual,  rtol=1e-3, atol=1)
                    self.assertAllClose(lh_expected, lh_actual, rtol=1e-3, atol=1)
                    m_train.train(sess)

                cs_expected = 68.57091859250194
                cs_actual = m_train.diagnose(sess)['curriculum_step']
                self.assertAllClose(cs_expected, cs_actual)
            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False) 

    def testTwoInvocationsFirstOrderLoss(self):
        # values sourced from single invocation runs
        with self.test_session(use_gpu=use_gpu) as sess:
            c_train = deepcopy(c_train_template)
            c_train.loss['tertiary_normalization'] = 'first'
            m_train = RGNModel('training', c_train)

            c_eval = deepcopy(c_eval_template)
            c_eval.loss['tertiary_normalization'] = 'first'
            c_eval.queueing['num_evaluation_invocations'] = 2
            c_eval.optimization['batch_size'] = eval_batch_size / c_eval.queueing['num_evaluation_invocations']
            m_eval = RGNModel('evaluation', c_eval)

            m_train.start([m_eval], sess, False)
            assign_weights(sess, w_template)

            try:
                for l_expected in [11.653967160378857, 29.682763504342415, 94.215383972698928]:
                    l_actual = m_eval.evaluate(sess)['tertiary_loss_all']
                    self.assertAllClose(l_expected, l_actual, rtol=5e-2, atol=5e-2)
                    m_train.train(sess)
            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False)  

    def testEvaluationSubgroupsZerothOrderLossWithFourInvocations(self):
        # sourced from single invocation runs
        train_files = ['1', '2', '3']
        eval_files  = ['1']
        train_dir = base_dir + 'data/unofficial/tfrecord/training_long_with_subgroups/'
        eval_dir  = base_dir + 'data/unofficial/tfrecord/test_long_with_subgroups/'
        state_size = 2
        batch_size = 224

        with self.test_session(use_gpu=use_gpu) as sess:
            c_train = deepcopy(c_train_template)
            c_train.io['data_files'] = [os.path.join(train_dir, file) for file in train_files]
            c_train.io['num_evo_entries'] = 42
            c_train.architecture['recurrent_layer_size'] = [state_size]
            c_train.optimization['batch_size'] = batch_size
            c_train.optimization['num_steps'] = 700
            m_train = RGNModel('training', c_train)

            c_eval = deepcopy(c_train)
            c_eval.io['data_files'] = [os.path.join(eval_dir, file) for file in eval_files]
            c_eval.io['evaluation_sub_groups'] = ['10', '20', '30', '40', '50', '70', '90']
            c_eval.queueing['num_evaluation_invocations'] = 4
            c_eval.optimization['batch_size'] = batch_size / c_eval.queueing['num_evaluation_invocations']
            c_eval.optimization['min_after_dequeue'] = 1
            m_eval = RGNModel('evaluation', c_eval)

            npr.seed(1)
            w = {'rnn/lstm_cell/kernel':    (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
                 'rnn/lstm_cell/bias':     (npr.rand(state_size * 4) - 0.5) * 0.15,
                 'rnn/lstm_cell/w_f_diag':   (npr.rand(state_size) - 0.5) * 0.05,
                 'rnn/lstm_cell/w_i_diag':   (npr.rand(state_size) - 0.5) * 0.05,
                 'rnn/lstm_cell/w_o_diag':   (npr.rand(state_size) - 0.5) * 0.05,
                 'linear_dihedrals/weights': (npr.rand(state_size, output_size) - 0.5) * 0.05,
                 'linear_dihedrals/biases':  ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

            m_train.start([m_eval], sess, False)
            assign_weights(sess, w)

            try:
                for i in range(3):
                    l_actual = m_eval.evaluate(sess)        
                    for subgroup, l_expected in {'all': [39.697769, 184.72499, 261.18362],
                                                 '10':  [25.821846, 114.27982, 163.8459],
                                                 '20':  [36.049255, 170.34642, 241.68335],
                                                 '30':  [39.677151, 187.0887,  264.8533],
                                                 '40':  [36.924805, 176.66263, 250.02922],
                                                 '50':  [45.720215, 212.67812, 299.69229],
                                                 '70':  [49.584141, 225.6004,  317.39334],
                                                 '90':  [44.106983, 206.41902, 290.78809]}.iteritems():
                        self.assertAllClose(l_expected[i], l_actual['tertiary_loss_' + subgroup], rtol=5e-2, atol=5e-2)        
                    m_train.train(sess)
            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False)  

    def testEvaluationSubgroupsFirstOrderLossWithFourInvocations(self):
        # sourced from single invocation runs
        train_files = ['1', '2', '3']
        eval_files  = ['1']
        train_dir = base_dir + 'data/unofficial/tfrecord/training_long_with_subgroups/'
        eval_dir  = base_dir + 'data/unofficial/tfrecord/test_long_with_subgroups/'
        state_size = 2
        batch_size = 224

        with self.test_session(use_gpu=use_gpu) as sess:
            c_train = deepcopy(c_train_template)
            c_train.io['data_files'] = [os.path.join(train_dir, file) for file in train_files]
            c_train.io['num_evo_entries'] = 42
            c_train.architecture['recurrent_layer_size'] = [state_size]
            c_train.optimization['batch_size'] = batch_size
            c_train.optimization['num_steps'] = 700
            c_train.loss['tertiary_normalization'] = 'first'
            m_train = RGNModel('training', c_train)

            c_eval = deepcopy(c_train)
            c_eval.io['data_files'] = [os.path.join(eval_dir, file) for file in eval_files]
            c_eval.io['evaluation_sub_groups'] = ['10', '20', '30', '40', '50', '70', '90']
            c_eval.queueing['num_evaluation_invocations'] = 4
            c_eval.optimization['batch_size'] = batch_size / c_eval.queueing['num_evaluation_invocations']
            c_eval.optimization['min_after_dequeue'] = 1
            m_eval = RGNModel('evaluation', c_eval)

            npr.seed(1)
            w = {'rnn/lstm_cell/kernel':    (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
                 'rnn/lstm_cell/bias':     (npr.rand(state_size * 4) - 0.5) * 0.15,
                 'rnn/lstm_cell/w_f_diag':   (npr.rand(state_size) - 0.5) * 0.05,
                 'rnn/lstm_cell/w_i_diag':   (npr.rand(state_size) - 0.5) * 0.05,
                 'rnn/lstm_cell/w_o_diag':   (npr.rand(state_size) - 0.5) * 0.05,
                 'linear_dihedrals/weights': (npr.rand(state_size, output_size) - 0.5) * 0.05,
                 'linear_dihedrals/biases':  ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

            m_train.start([m_eval], sess, False)
            assign_weights(sess, w)

            try:
                for i in range(3):
                    l_actual = m_eval.evaluate(sess)        
                    for subgroup, l_expected in {'all': [60.723743, 376.06458, 270.76132],
                                                 '10':  [44.887554, 282.67255, 199.71959],
                                                 '20':  [51.896362, 331.10999, 236.40175],
                                                 '30':  [56.237797, 355.08456, 254.8051],
                                                 '40':  [50.637138, 323.91776, 231.41081],
                                                 '50':  [64.921356, 397.58215, 287.76514],
                                                 '70':  [74.773926, 452.44659, 328.10721],
                                                 '90':  [70.957344, 429.28339, 311.2869]}.iteritems():
                        self.assertAllClose(l_expected[i], l_actual['tertiary_loss_' + subgroup], rtol=5e-2, atol=5e-2)        
                    m_train.train(sess)
            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False)  

    def testEvaluationSubgroupsSecondOrderLossWithFourInvocations(self):
        # sourced from single invocation runs
        train_files = ['1', '2', '3']
        eval_files  = ['1']
        train_dir = base_dir + 'data/unofficial/tfrecord/training_long_with_subgroups/'
        eval_dir  = base_dir + 'data/unofficial/tfrecord/test_long_with_subgroups/'
        state_size = 2
        batch_size = 224

        with self.test_session(use_gpu=use_gpu) as sess:
            c_train = deepcopy(c_train_template)
            c_train.io['data_files'] = [os.path.join(train_dir, file) for file in train_files]
            c_train.io['num_evo_entries'] = 42
            c_train.architecture['recurrent_layer_size'] = [state_size]
            c_train.optimization['batch_size'] = batch_size
            c_train.optimization['num_steps'] = 700
            c_train.loss['tertiary_normalization'] = 'second'
            m_train = RGNModel('training', c_train)

            c_eval = deepcopy(c_train)
            c_eval.io['data_files'] = [os.path.join(eval_dir, file) for file in eval_files]
            c_eval.io['evaluation_sub_groups'] = ['10', '20', '30', '40', '50', '70', '90']
            c_eval.queueing['num_evaluation_invocations'] = 4
            c_eval.optimization['batch_size'] = batch_size / c_eval.queueing['num_evaluation_invocations']
            c_eval.optimization['min_after_dequeue'] = 1
            m_eval = RGNModel('evaluation', c_eval)

            npr.seed(1)
            w = {'rnn/lstm_cell/kernel':    (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
                 'rnn/lstm_cell/bias':     (npr.rand(state_size * 4) - 0.5) * 0.15,
                 'rnn/lstm_cell/w_f_diag':   (npr.rand(state_size) - 0.5) * 0.05,
                 'rnn/lstm_cell/w_i_diag':   (npr.rand(state_size) - 0.5) * 0.05,
                 'rnn/lstm_cell/w_o_diag':   (npr.rand(state_size) - 0.5) * 0.05,
                 'linear_dihedrals/weights': (npr.rand(state_size, output_size) - 0.5) * 0.05,
                 'linear_dihedrals/biases':  ((npr.rand(output_size) - 0.5) * 0.05) + 0.2}

            m_train.start([m_eval], sess, False)
            assign_weights(sess, w)

            try:
                for i in range(3):
                    l_actual = m_eval.evaluate(sess)        
                    for subgroup, l_expected in {'all': [78.809792, 297.50754, 186.81566],
                                                 '10':  [64.595146, 248.27773, 154.61333],
                                                 '20':  [64.715256, 251.11797, 156.10262],
                                                 '30':  [68.300896, 264.15610, 164.73335],
                                                 '40':  [63.078617, 247.08490, 153.70581],
                                                 '50':  [81.458519, 303.73416, 191.37691],
                                                 '70':  [92.466843, 344.84088, 217.63127],
                                                 '90':  [94.226669, 346.87625, 219.36301]}.iteritems():
                        self.assertAllClose(l_expected[i], l_actual['tertiary_loss_' + subgroup], rtol=5e-2, atol=5e-2)        
                    m_train.train(sess)
            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False)  

    def testLossChangeLossCurriculumAndHistoryUpdatingFirstOrder(self):
        train_files = ['1', '2', '3']
        eval_files  = ['1']
        train_dir = base_dir + 'data/unofficial/tfrecord/training_long_with_subgroups/'
        eval_dir  = base_dir + 'data/unofficial/tfrecord/test_long_with_subgroups/'
        state_size = 2
        batch_size = 224

        with self.test_session(use_gpu=use_gpu) as sess:
            c_train = deepcopy(c_train_template)
            c_train.io['data_files'] = [os.path.join(train_dir, file) for file in train_files]
            c_train.io['num_evo_entries'] = 42
            c_train.architecture['recurrent_layer_size'] = [state_size]
            c_train.optimization['batch_size'] = batch_size
            c_train.optimization['num_steps'] = 700
            c_train.loss['tertiary_normalization'] = 'first'
            c_train.curriculum['mode'] = 'loss'
            c_train.curriculum['behavior'] = 'loss_change'
            c_train.curriculum['slope'] = 1.0
            c_train.curriculum['base'] = 40.0
            c_train.curriculum['rate'] = 10.0
            c_train.curriculum['change_num_iterations'] = 5
            c_train.curriculum['sharpness'] = 0.1
            m_train = RGNModel('training', c_train)

            c_eval = deepcopy(c_train)
            c_eval.io['data_files'] = [os.path.join(eval_dir, file) for file in eval_files]
            c_eval.curriculum['update_loss_history'] = True
            m_eval = RGNModel('evaluation', c_eval)

            npr.seed(1)
            w = {'rnn/lstm_cell/kernel':    (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
                 'rnn/lstm_cell/bias':     (npr.rand(state_size * 4) - 0.5) * 0.15,
                 'rnn/lstm_cell/w_f_diag':   (npr.rand(state_size) - 0.5) * 0.05,
                 'rnn/lstm_cell/w_i_diag':   (npr.rand(state_size) - 0.5) * 0.05,
                 'rnn/lstm_cell/w_o_diag':   (npr.rand(state_size) - 0.5) * 0.05,
                 'linear_dihedrals/weights': (npr.rand(state_size, output_size) - 0.5) * 0.05,
                 'linear_dihedrals/biases':  ((npr.rand(output_size) - 0.5) * 0.05) + 0.2,
                 'curriculum_loss_history':  np.array([-1., -1, 10., 10., 10.])}

            m_train.start([m_eval], sess, False)
            assign_weights(sess, w)

            try:
                for l_expected, lh_expected in zip([10.549246, 34.995762, 36.922672, 75.110641, 25.719475],
                                                   [[ -1.        ,  10.        ,  10.        ,  10.        ,  10.54924583], 
                                                    [ 10.        ,  10.        ,  10.        ,  10.54924583,  34.99576187], 
                                                    [ 10.        ,  10.        ,  10.54924583,  34.99576187,  36.92267227], 
                                                    [ 10.        ,  10.54924583,  34.99576187,  36.92267227,  75.11064148], 
                                                    [ 10.54924583,  34.99576187,  36.92267227,  75.11064148,  25.71947479]]):
                    eval_dict = m_eval.evaluate(sess, pretty=False) 
                    l_actual = eval_dict['tertiary_loss_all']
                    lh_actual = eval_dict['update_curriculum_history_op']
                    self.assertAllClose(l_expected,  l_actual,  rtol=1e-3, atol=1)
                    self.assertAllClose(lh_expected, lh_actual, rtol=1e-3, atol=1)
                    m_train.train(sess)

                cs_expected = 68.164871
                cs_actual = m_train.diagnose(sess)['curriculum_step']
                self.assertAllClose(cs_expected, cs_actual)
            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False) 

    def testLossChangeLossCurriculumAndHistoryUpdatingFirstOrderWithEvaluationSubgroupsAndFourInvocations(self):
        train_files = ['1', '2', '3']
        eval_files  = ['1']
        train_dir = base_dir + 'data/unofficial/tfrecord/training_long_with_subgroups/'
        eval_dir  = base_dir + 'data/unofficial/tfrecord/test_long_with_subgroups/'
        state_size = 2
        batch_size = 224

        with self.test_session(use_gpu=use_gpu) as sess:
            c_train = deepcopy(c_train_template)
            c_train.io['data_files'] = [os.path.join(train_dir, file) for file in train_files]
            c_train.io['num_evo_entries'] = 42
            c_train.architecture['recurrent_layer_size'] = [state_size]
            c_train.optimization['batch_size'] = batch_size
            c_train.optimization['num_steps'] = 700
            c_train.loss['tertiary_normalization'] = 'first'
            c_train.curriculum['mode'] = 'loss'
            c_train.curriculum['behavior'] = 'loss_change'
            c_train.curriculum['slope'] = 1.0
            c_train.curriculum['base'] = 40.0
            c_train.curriculum['rate'] = 10.0
            c_train.curriculum['change_num_iterations'] = 5
            c_train.curriculum['sharpness'] = 0.1
            m_train = RGNModel('training', c_train)

            c_eval = deepcopy(c_train)
            c_eval.io['data_files'] = [os.path.join(eval_dir, file) for file in eval_files]
            c_eval.io['evaluation_sub_groups'] = ['10', '20', '30', '40', '50', '70', '90']
            c_eval.queueing['num_evaluation_invocations'] = 4
            c_eval.optimization['batch_size'] = batch_size / c_eval.queueing['num_evaluation_invocations']
            c_eval.curriculum['update_loss_history'] = True
            m_eval = RGNModel('evaluation', c_eval)

            npr.seed(1)
            w = {'rnn/lstm_cell/kernel':    (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
                 'rnn/lstm_cell/bias':     (npr.rand(state_size * 4) - 0.5) * 0.15,
                 'rnn/lstm_cell/w_f_diag':   (npr.rand(state_size) - 0.5) * 0.05,
                 'rnn/lstm_cell/w_i_diag':   (npr.rand(state_size) - 0.5) * 0.05,
                 'rnn/lstm_cell/w_o_diag':   (npr.rand(state_size) - 0.5) * 0.05,
                 'linear_dihedrals/weights': (npr.rand(state_size, output_size) - 0.5) * 0.05,
                 'linear_dihedrals/biases':  ((npr.rand(output_size) - 0.5) * 0.05) + 0.2,
                 'curriculum_loss_history':  np.array([-1., -1, 10., 10., 10.])}

            m_train.start([m_eval], sess, False)
            assign_weights(sess, w)

            try:
                for l_expected, lh_expected in zip([10.549246, 34.995762, 36.922672, 75.110641, 25.719475],
                                                   [[ -1.        ,  10.        ,  10.        ,  10.        ,  10.54924583], 
                                                    [ 10.        ,  10.        ,  10.        ,  10.54924583,  34.99576187], 
                                                    [ 10.        ,  10.        ,  10.54924583,  34.99576187,  36.92267227], 
                                                    [ 10.        ,  10.54924583,  34.99576187,  36.92267227,  75.11064148], 
                                                    [ 10.54924583,  34.99576187,  36.92267227,  75.11064148,  25.71947479]]):
                    eval_dict = m_eval.evaluate(sess, pretty=False) 
                    l_actual = eval_dict['tertiary_loss_all']
                    lh_actual = eval_dict['update_curriculum_history_op']
                    self.assertAllClose(l_expected,  l_actual,  rtol=1e-3, atol=1)
                    self.assertAllClose(lh_expected, lh_actual, rtol=1e-3, atol=1)
                    m_train.train(sess)

                cs_expected = 68.164871
                cs_actual = m_train.diagnose(sess)['curriculum_step']
                self.assertAllClose(cs_expected, cs_actual)
            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False) 

    def testLossChangeLossCurriculumAndHistoryUpdatingFirstOrderWithEvaluationSubgroupsFourInvocationsAndMultipleEvaluationModels(self):
        train_files = ['1', '2', '3']
        eval_files  = ['1']
        train_dir = base_dir + 'data/unofficial/tfrecord/training_long_with_subgroups/'
        eval_dir  = base_dir + 'data/unofficial/tfrecord/test_long_with_subgroups/'
        logs_dir  = base_dir + 'checkpoints/'
        state_size = 2
        batch_size = 224

        with self.test_session(use_gpu=use_gpu) as sess:
            c_train = deepcopy(c_train_template)
            c_train.io['data_files'] = [os.path.join(train_dir, file) for file in train_files]
            c_train.io['num_evo_entries'] = 42
            c_train.io['logs_directory'] = logs_dir
            c_train.io['log_model_summaries'] = True
            c_train.io['detailed_logs'] = True
            c_train.architecture['recurrent_layer_size'] = [state_size]
            c_train.optimization['batch_size'] = batch_size
            c_train.optimization['num_steps'] = 700
            c_train.optimization['learning_rate'] = 0.
            c_train.loss['tertiary_normalization'] = 'first'
            c_train.curriculum['mode'] = 'loss'
            c_train.curriculum['behavior'] = 'loss_change'
            c_train.curriculum['slope'] = 1.0
            c_train.curriculum['base'] = 40.0
            c_train.curriculum['rate'] = 10.0
            c_train.curriculum['change_num_iterations'] = 5
            c_train.curriculum['sharpness'] = 0.1
            m_train = RGNModel('training', c_train)

            c_eval = deepcopy(c_train)
            c_eval.io['data_files'] = [os.path.join(eval_dir, file) for file in eval_files]
            c_eval.io['evaluation_sub_groups'] = ['10', '20', '30', '40', '50', '70', '90']
            c_eval.queueing['num_evaluation_invocations'] = 4
            c_eval.optimization['batch_size'] = batch_size / c_eval.queueing['num_evaluation_invocations']
            c_eval.curriculum['update_loss_history'] = True
            m_eval = RGNModel('evaluation', c_eval)

            c_unwt_eval = deepcopy(c_eval)
            c_unwt_eval.curriculum['update_loss_history'] = False
            c_unwt_eval.curriculum['mode'] = None
            c_unwt_eval.curriculum['behavior'] = None
            m_unwt_eval = RGNModel('evaluation', c_unwt_eval)

            npr.seed(1)
            w = {'rnn/lstm_cell/kernel':    (npr.rand(input_size + state_size, state_size * 4) - 0.5) * 0.05,
                 'rnn/lstm_cell/bias':     (npr.rand(state_size * 4) - 0.5) * 0.15,
                 'rnn/lstm_cell/w_f_diag':   (npr.rand(state_size) - 0.5) * 0.05,
                 'rnn/lstm_cell/w_i_diag':   (npr.rand(state_size) - 0.5) * 0.05,
                 'rnn/lstm_cell/w_o_diag':   (npr.rand(state_size) - 0.5) * 0.05,
                 'linear_dihedrals/weights': (npr.rand(state_size, output_size) - 0.5) * 0.05,
                 'linear_dihedrals/biases':  ((npr.rand(output_size) - 0.5) * 0.05) + 0.2,
                 'curriculum_loss_history':  np.array([-1., -1, 10., 10., 10.])}

            m_train.start([m_eval, m_unwt_eval], sess, False)
            assign_weights(sess, w)

            try:
                for l_expected, l_unwt_expected in zip([10.549246, 10.549246, 10.908124],
                                                       [60.723743, 60.723743, 60.723743]):
                    
                    eval_dict = m_eval.evaluate(sess, pretty=False)
                    m_train.diagnose(sess)
                    unwt_eval_dict = m_unwt_eval.evaluate(sess, pretty=False)

                    l_actual      = eval_dict['tertiary_loss_all']
                    l_unwt_actual = unwt_eval_dict['tertiary_loss_all']
                    self.assertAllClose(l_expected,      l_actual,      rtol=1e-3, atol=1)
                    self.assertAllClose(l_unwt_expected, l_unwt_actual, rtol=1e-3, atol=1)

                    m_train.train(sess)
            finally:
                m_train.finish(sess, save=False, close_session=False, reset_graph=False) 


# run tests
if __name__ == "__main__":
    tf.test.main()
