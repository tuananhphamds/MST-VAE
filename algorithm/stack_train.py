# -*- coding: utf-8 -*-
import os
import sys
import logging
import time
import numpy as np
import tensorflow as tf
import tfsnippet as spt
import mltk
import json

from tfsnippet.scaffold import TrainLoop
from tfsnippet.trainer import Trainer, Evaluator

HOME = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(HOME + '/algorithm')
sys.path.append(HOME + '/explib')


from utils import get_data_dim, get_data, get_data_name, get_sliding_window_data_flow, time_generator, GraphNodes
from MST_VAE import ModelConfig, MTSAD
from stack_predict import PredictConfig


class TrainConfig(mltk.Config):
    # training params
    batch_size = 100
    max_epoch = 40 # 30-20
    train_start = 0
    max_train_size = None  # `None` means full train set
    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 30
    lr_anneal_step_freq = None

    early_stopping = True
    valid_portion = 0.3


class ExpConfig(mltk.Config):
    seed = int(time.time())

    dataset = 'omi-1'

    # model params
    model = ModelConfig()

    @mltk.root_checker()
    def _model_post_checker(self, v: 'ExpConfig'):
        if v.model.x_dim == -1:
            v.model.x_dim = get_data_dim(v.dataset)

    use_time_info = False  # whether to use time information (minute, hour, day) as input u. discarded.

    model_type = 'mtsad'

    # train params
    train = TrainConfig()

    @mltk.root_checker()
    def _train_post_checker(self, v: 'ExpConfig'):
        if v.dataset == 'SWaT' or v.dataset == 'WADI':
            v.train.max_epoch = 15
            v.train.pretrain_max_epoch = 10
            v.train.pretrain_lr_anneal_epoch_freq = 5
            v.train.lr_anneal_epoch_freq = 5
        if v.dataset == 'SWaT':
            v.train.initial_lr = 0.0005
        if v.dataset == 'WADI':
            v.train.initial_lr = 0.0002

    test = PredictConfig()

    # debugging params
    write_summary = False
    write_histogram_summary = False
    check_numerics = False
    save_results = True
    save_ckpt = True
    ckpt_epoch_freq = 10
    ckpt_max_keep = 10
    pretrain_ckpt_epoch_freq = 20
    pretrain_ckpt_max_keep = 10

    exp_dir_save_path = None    # The file path to save the exp dirs for batch run training on different datasets.


def get_lr_value(init_lr,
                 anneal_factor,
                 anneal_freq,
                 loop: spt.TrainLoop,
                 ) -> spt.DynamicValue:
    """
    Get the learning rate scheduler for specified experiment.

    Args:
        exp: The experiment object.
        loop: The train loop object.

    Returns:
        A dynamic value, which returns the learning rate each time
        its `.get()` is called.
    """
    return spt.AnnealingScalar(
        loop=loop,
        initial_value=init_lr,
        ratio=anneal_factor,
        epochs=anneal_freq,
    )

def train_sgvb_loss(qnet, pnet, metrics_dict: GraphNodes, prefix='pretrain_', name=None):
    with tf.name_scope(name, default_name='pre_sgvb_loss'):
        logpx_z = pnet['x'].log_prob(name='logpx_z')
        logqz_x = qnet['z'].log_prob(name='logpz_x')
        logpz = pnet['z'].log_prob(name='logpz')

        kl_term = tf.reduce_mean(logqz_x - logpz)
        recons_term = tf.reduce_mean(logpx_z)
        metrics_dict[prefix + 'recons'] = recons_term
        metrics_dict[prefix + 'kl'] = kl_term

        return -tf.reduce_mean(logpx_z + 0.2 * (logpz - logqz_x))


def main(exp: mltk.Experiment[ExpConfig], config: ExpConfig):
    logging.basicConfig(
        level='INFO',
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    # print the current seed and generate three seeds
    logging.info('Current random seed: %s', config.seed)
    np.random.seed(config.seed)

    spt.settings.check_numerics = config.check_numerics
    spt.settings.enable_assertions = False

    # print the config
    print(mltk.format_key_values(config, title='Configurations'))
    print('')

    # open the result object and prepare for result directories
    exp.make_dirs('train_summary')
    exp.make_dirs('result_params')
    exp.make_dirs('ckpt_params')
    exp.make_dirs(config.test.output_dirs)

    # prepare the data
    # simple data
    (x_train, _), (x_test, y_test) = \
        get_data(config.dataset, config.train.max_train_size, config.test.max_test_size,
                 train_start=config.train.train_start, test_start=config.test.test_start,
                 valid_portion=config.train.valid_portion)

    if config.use_time_info:
        u_train = np.asarray([time_generator(_i) for _i in range(len(x_train))])  # (train_size, u_dim)
        u_test = np.asarray([time_generator(len(x_train) + _i) for _i in range(len(x_test))])  # (test_size, u_dim)
    else:
        u_train = np.zeros([len(x_train), config.model.u_dim])  # (train_size, u_dim)
        u_test = np.zeros([len(x_test), config.model.u_dim])

    split_idx = int(len(x_train) * config.train.valid_portion)
    x_train, x_valid = x_train[:-split_idx], x_train[-split_idx:]
    u_train, u_valid = u_train[:-split_idx], u_train[-split_idx:]

    # prepare data_flow
    train_flow = get_sliding_window_data_flow(window_size=config.model.window_length,
                                              batch_size=config.train.batch_size,
                                              x=x_train, u=u_train, shuffle=True, skip_incomplete=True)
    valid_flow = get_sliding_window_data_flow(window_size=config.model.window_length,
                                              batch_size=config.train.batch_size,
                                              x=x_valid, u=u_valid, shuffle=False, skip_incomplete=False)

    # build computation graph
    model = MTSAD(config.model, scope='model')

    # input placeholders
    input_x = tf.placeholder(dtype=tf.float32, shape=[None, config.model.window_length, config.model.x_dim],
                             name='input_x')
    input_u = tf.placeholder(dtype=tf.float32, shape=[None, config.model.window_length, config.model.u_dim],
                             name='input_u')
    learning_rate = tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
    is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

    # derive training nodes
    with tf.name_scope('training'):
        # train the whole network with z1 and z2
        train_q_net = model.q_net(input_x, u=input_u, is_training=is_training)
        train_chain = train_q_net.chain(model.p_net, observed={'x': input_x}, u=input_u, is_training=is_training)
        train_metrics = GraphNodes()
        vae_loss = train_sgvb_loss(train_chain.variational, train_chain.model, train_metrics, name='train_sgvb_loss')
        reg_loss = tf.losses.get_regularization_loss()
        loss = vae_loss + reg_loss
        train_metrics['loss'] = loss

    with tf.name_scope('validation'):
        # validation of the whole network
        valid_q_net = model.q_net(input_x, u=input_u, n_z=config.test.test_n_z)
        valid_chain = valid_q_net.chain(model.p_net, observed={'x': input_x}, latent_axis=0, u=input_u)
        valid_metrics = GraphNodes()
        valid_loss = train_sgvb_loss(valid_chain.variational, valid_chain.model, valid_metrics, prefix='valid_',
                               name='valid_sgvb_loss') + tf.losses.get_regularization_loss()
        valid_metrics['valid_loss'] = valid_loss

    # obtain params and gradients (whole model)
    variables_to_save = tf.global_variables()
    train_params = tf.trainable_variables()

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gradients = optimizer.compute_gradients(loss, var_list=train_params)
    # clip gradient by norm
    with tf.name_scope('ClipGradients'):
        for i, (g, v) in enumerate(gradients):
            if g is not None:
                gradients[i] = (tf.clip_by_norm(
                    spt.utils.maybe_check_numerics(g, message="gradient on %s exceed" % str(v.name)), 10), v)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.apply_gradients(gradients)

    var_groups = [
        # for q_net
        model.variable_scope.name + '/q_net',

        # for p_net
        model.variable_scope.name + '/p_net',

        # for flow
        model.variable_scope.name + '/posterior_flow'
    ]

    var_initializer = tf.variables_initializer(tf.global_variables())

    train_flow = train_flow.threaded(5)
    valid_flow = valid_flow.threaded(5)

    loop = TrainLoop(param_vars=variables_to_save,
                     var_groups=var_groups,
                     max_epoch=config.train.max_epoch,
                     summary_dir=(exp.abspath('train_summary')
                                  if config.write_summary else None),
                     summary_graph=tf.get_default_graph(),
                     summary_commit_freqs={'loss': 10},
                     early_stopping=config.train.early_stopping,
                     valid_metric_name='valid_loss',
                     valid_metric_smaller_is_better=True,
                     checkpoint_dir=(exp.abspath('ckpt_params')
                                     if config.save_ckpt else None),
                     checkpoint_epoch_freq=config.ckpt_epoch_freq,
                     checkpoint_max_to_keep=config.ckpt_max_keep
                     )

    if config.write_histogram_summary:
        summary_op = tf.summary.merge_all()
    else:
        summary_op = None

    lr_value = get_lr_value(config.train.initial_lr, config.train.lr_anneal_factor,
                            config.train.lr_anneal_epoch_freq, loop)

    trainer = Trainer(loop=loop,
                      train_op=train_op,
                      inputs=[input_x, input_u],
                      data_flow=train_flow,
                      feed_dict={learning_rate: lr_value, is_training: True},
                      metrics=train_metrics,
                      summaries=summary_op)

    validator = Evaluator(loop=loop,
                          metrics=valid_metrics,
                          inputs=[input_x, input_u],
                          data_flow=valid_flow,
                          time_metric_name='valid_time')

    validator.events.on(
        spt.EventKeys.AFTER_EXECUTION,
        lambda e: exp.update_results(validator.last_metrics_dict)
    )

    train_losses = []
    tmp_collector = []
    valid_losses = []

    def on_metrics_collected(loop: TrainLoop, metrics):
        if 'loss' in metrics:
            tmp_collector.append(metrics['loss'])
        if loop.epoch % 1 == 0:
            if 'valid_loss' in metrics:
                valid_losses.append(metrics['valid_loss'])
                train_losses.append(np.mean(tmp_collector))
                tmp_collector.clear()

    loop.events.on(spt.EventKeys.METRICS_COLLECTED, on_metrics_collected)

    trainer.evaluate_after_epochs(validator, freq=1)
    trainer.log_after_epochs(freq=1)

    with spt.utils.create_session().as_default() as session:

        session.run(var_initializer)

        print('************Start train the whole network***********')

        with loop:
            trainer.run()

        print('')
        print('Training Finished.')

        if config.save_results:
            saver = tf.train.Saver(var_list=variables_to_save)
            saver.save(session, os.path.join(exp.abspath('result_params'), "restored_params.dat"))

        print('')
        print('Model saved.')


def run_train(dataset):
    data_name = get_data_name(dataset)
    with open('algorithm/train_config.json', 'r') as f:
        train_cfg = json.load(f)[data_name]
        print(train_cfg)

    with mltk.Experiment(ExpConfig()) as exp:
        exp.config.dataset = dataset
        if dataset == "SWaT":
            exp.config.train.train_start = 21600
            exp.config.model.window_length = 30
        elif dataset == "WADI":
            exp.config.train.train_start = 259200
            exp.config.train.max_train_size = 789371

        exp.config.train.max_epoch = train_cfg['max_epoch']
        exp.config.train.initial_lr = train_cfg['initial_lr']
        exp.config.train.lr_anneal_epoch_freq = train_cfg['lr_anneal_epoch_freq']
        exp.config.train.valid_portion = train_cfg['valid_portion']
        exp.config.model.window_length = train_cfg['window_length']
        exp.config.model.output_shape = train_cfg['output_shape']
        exp.config.model.z_dim = train_cfg['z_dim']

        exp.config.model.x_dim = get_data_dim(dataset)
        exp.save_config()
        main(exp, exp.config)
        if exp.config.exp_dir_save_path is not None:
            with open(exp.config.exp_dir_save_path, 'a') as f:
                f.write("'" + exp.config.dataset + ' ' + exp.output_dir + "'" + '\n')


if __name__ == '__main__':
    run_train()
