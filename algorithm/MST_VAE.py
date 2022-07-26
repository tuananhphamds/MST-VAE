from enum import Enum
from typing import Optional, List

import logging
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tfsnippet.bayes import BayesianNet
from tfsnippet.utils import (instance_reuse,
                             VarScopeObject,
                             reopen_variable_scope)
from tfsnippet.distributions import FlowDistribution, Normal
from tfsnippet.layers import l2_regularizer
import mltk
from conv1d_ import conv1d, deconv1d


class RNNCellType(str, Enum):
    GRU = 'GRU'
    LSTM = 'LSTM'
    Basic = 'Basic'


class ModelConfig(mltk.Config):
    x_dim: int = -1
    z_dim: int = 3
    u_dim: int = 1
    window_length = 100
    output_shape: List[int] = [25, 50, 100]
    # output_shape: List[int] = [15, 30]
    z_dim: int = 13
    l2_reg = 0.0001
    unified_px_logstd = False

    logstd_min = -5.
    logstd_max = 2.
    use_prior_flow = False  # If True, use RealNVP prior flow to enhance the representation of p(z).
    connect_qz = True
    connect_pz = True

class MTSAD(VarScopeObject):

    def __init__(self, config: ModelConfig, name=None, scope=None):
        self.config = config
        super(MTSAD, self).__init__(name=name, scope=scope)

    def reconstruct(self, x, u, mask, n_z=None):
        with tf.name_scope('model.reconstruct'):
            qnet = self.q_net(x=x, u=u, n_z=n_z)
            pnet = self.p_net(observed={'z': qnet['z']}, u=u)
        return pnet['x']

    def get_score(self, x_embed, x_eval, u, n_z=None):
        with tf.name_scope('model.get_score'):
            qnet = self.q_net(x=x_embed, u=u, n_z=n_z)
            pnet = self.p_net(observed={'z': qnet['z']}, u=u)
            score = pnet['x'].distribution.base_distribution.log_prob(x_eval)
            recons_mean = pnet['x'].distribution.base_distribution.mean
            recons_std = pnet['x'].distribution.base_distribution.std
            if n_z is not None:
                score = tf.reduce_mean(score, axis=0)
                recons_mean = tf.reduce_mean(recons_mean, axis=0)
                recons_std = tf.reduce_mean(recons_std, axis=0)
        return score, recons_mean, recons_std

    @instance_reuse
    def q_net(self, x, observed=None, u=None, n_z=None, is_training=False):
        logging.info('pretrain_q_net builder: %r', locals())

        net = BayesianNet(observed=observed)

        def dropout_fn(input):
            return tf.layers.dropout(input, rate=.5, training=is_training)

        qz_mean, qz_logstd = self.h_for_qz(x)

        qz_distribution = Normal(mean=qz_mean, logstd=qz_logstd)

        qz_distribution = qz_distribution.batch_ndims_to_value(2)

        z = net.add('z', qz_distribution, n_samples=n_z,
                    is_reparameterized=True)

        return net

    @instance_reuse
    def p_net(self, observed=None, u=None, n_z=None, is_training=False):
        logging.info('pretrain p_net builder: %r', locals())

        net = BayesianNet(observed=observed)

        pz_distribution = Normal(mean=tf.zeros([self.config.z_dim, self.config.x_dim]),
                                 logstd=tf.zeros([self.config.z_dim, self.config.x_dim]))

        pz_distribution = pz_distribution.batch_ndims_to_value(2)

        z = net.add('z',
                    pz_distribution,
                    n_samples=n_z, is_reparameterized=True)

        h_z = self.h_for_px(z)

        px_mean = conv1d(h_z, kernel_size=1, out_channels=self.config.x_dim, scope='pre_px_mean')
        px_logstd = conv1d(h_z, kernel_size=1, out_channels=self.config.x_dim, scope='pre_px_logstd')
        px_logstd = tf.clip_by_value(px_logstd, clip_value_min=self.config.logstd_min,
                                     clip_value_max=self.config.logstd_max)

        x = net.add('x',
                    Normal(mean=px_mean, logstd=px_logstd).batch_ndims_to_value(2),
                    is_reparameterized=True)

        return net

    @instance_reuse
    def h_for_qz(self, x):
        output_shape = self.config.output_shape
        with arg_scope([conv1d],
                       kernel_size=2,
                       activation_fn=tf.nn.relu,
                       kernel_regularizer=l2_regularizer(self.config.l2_reg),
                       out_channels=self.config.x_dim,
                       strides=2):
            # Extract features from short-scale module
            h_x1 = None
            for i in range(len(output_shape)):
                if h_x1 is None:
                    h_x1 = conv1d(x)
                else:
                    h_x1 = conv1d(h_x1)

            # Extract features from long-scale module
            h_x2 = None
            for i in range(len(output_shape)):
                if h_x2 is None:
                    h_x2 = conv1d(x, kernel_size=15)
                else:
                    h_x2 = conv1d(h_x2)

            if h_x1 is None or h_x2 is None:
                raise Exception('Hidden features cannot be None, check output shape')

            # Concatenate layer
            concat = tf.concat([h_x1, h_x2], axis=-1)

            # Feature reduction layer
            conv_concat = conv1d(concat, out_channels=self.config.x_dim, strides=1, kernel_size=1)

        # Mean and log standard deviation layers
        qz_mean = conv1d(conv_concat, kernel_size=1, out_channels=self.config.x_dim)
        qz_logstd = conv1d(conv_concat, kernel_size=1, out_channels=self.config.x_dim)
        qz_logstd = tf.clip_by_value(qz_logstd, clip_value_min=self.config.logstd_min,
                                     clip_value_max=self.config.logstd_max)
        return qz_mean, qz_logstd

    @instance_reuse
    def h_for_px(self, z):
        output_shape = self.config.output_shape
        with arg_scope([deconv1d],
                       kernel_size=2,
                       strides=2,
                       activation_fn=tf.nn.relu,
                       kernel_regularizer=l2_regularizer(self.config.l2_reg),
                       out_channels=self.config.x_dim):

            # Reconstruct input from short-scale features
            h_z1 = deconv1d(z, output_shape=self.config.z_dim, strides=1, kernel_size=1)
            for i in range(len(output_shape)):
                h_z1 = deconv1d(h_z1, output_shape=output_shape[i])

            # Reconstruct input from long-scale features
            h_z2 = deconv1d(z, output_shape=self.config.z_dim, strides=1, kernel_size=1)
            for i in range(len(output_shape)):
                if i == len(output_shape) - 1:
                    h_z2 = deconv1d(h_z2, output_shape=output_shape[i], kernel_size=15)
                else:
                    h_z2 = deconv1d(h_z2, output_shape=output_shape[i])

            # Concatenate layer
            concat = tf.concat([h_z1, h_z2], axis=-1)

            # Feature reduction layer
            h_z = deconv1d(concat, out_channels=self.config.x_dim, output_shape=output_shape[-1], strides=1,
                        kernel_size=1)
        return h_z
