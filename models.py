import tensorflow as tf

import ops
from datahandler import datashapes


def encoder(opts, inputs, reuse=tf.AUTO_REUSE, is_training=False, y=None):
    with tf.variable_scope("encoder", reuse=reuse):
        return dcgan_encoder(opts, inputs, is_training, reuse, y)


def classifier(opts, noise, reuse=tf.AUTO_REUSE):
    with tf.variable_scope("classifier", reuse=reuse):
        if opts['mlp_classifier']:
            out = ops.linear(opts, noise, 500, 'mlp1')
            out = tf.nn.relu(out)
            out = ops.linear(opts, out, 500, 'mlp2')
            out = tf.nn.relu(out)
            logits = ops.linear(opts, out, opts['n_classes'], 'classifier')
        else:
            logits = ops.linear(opts, noise, opts['n_classes'], 'classifier')
    return logits


def decoder(opts, noise, reuse=tf.AUTO_REUSE, is_training=True):
    with tf.variable_scope("generator", reuse=reuse):
        res = dcgan_decoder(opts, noise, is_training, reuse)
        return res


def dcgan_encoder(opts, inputs, is_training=False, reuse=False, y=None):
    num_units = opts['e_num_filters']
    num_layers = opts['e_num_layers']
    layer_x = inputs
    for i in range(num_layers):
        scale = 2 ** (num_layers - i - 1)
        layer_x = ops.conv2d(opts, layer_x, num_units / scale,
                             scope='h%d_conv' % i)
        if opts['batch_norm']:
            layer_x = ops.batch_norm(opts, layer_x, is_training,
                                     reuse, scope='h%d_bn' % i)
        layer_x = tf.nn.relu(layer_x)
    else:
        h_dim = opts['hdim']
        h = tf.reshape(layer_x, [-1, h_dim])
        y_onehot = tf.one_hot(y, opts['n_classes'])
        h_y = tf.concat((h, y_onehot), axis=1)
        h = ops.linear(opts, h_y, h_dim, scope='h_y_lin')
        mean = ops.linear(opts, h, opts['zdim'], scope='mean_lin')
        log_sigmas = ops.linear(opts, h,
                                opts['zdim'], scope='log_sigmas_lin')
        return mean, log_sigmas


def dcgan_decoder(opts, noise, is_training=False, reuse=False):
    output_shape = datashapes[opts['dataset']]
    num_units = opts['g_num_filters']
    batch_size = tf.shape(noise)[0]
    num_layers = opts['g_num_layers']
    height = output_shape[0] // 2 ** (num_layers - 1)
    width = output_shape[1] // 2 ** (num_layers - 1)

    h0 = ops.linear(
        opts, noise, num_units * height * width, scope='h0_lin')
    h0 = tf.reshape(h0, [-1, height, width, num_units])
    h0 = tf.nn.relu(h0)
    layer_x = h0
    for i in range(num_layers - 1):
        scale = 2 ** (i + 1)
        _out_shape = [batch_size, height * scale,
                      width * scale, num_units // scale]
        layer_x = ops.deconv2d(opts, layer_x, _out_shape,
                               scope='h%d_deconv' % i)
        if opts['batch_norm']:
            layer_x = ops.batch_norm(opts, layer_x,
                                     is_training, reuse, scope='h%d_bn' % i)
        layer_x = tf.nn.relu(layer_x)
    _out_shape = [batch_size] + list(output_shape)

    last_h = ops.deconv2d(
        opts, layer_x, _out_shape, d_h=1, d_w=1, scope='hfinal_deconv')
    return tf.nn.sigmoid(last_h)
