import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import sys
sys.path.append('/home/maurice/mmd')
import tensorflow as tf
layers = tf.layers

from matplotlib.gridspec import GridSpec
from tensorflow.examples.tutorials.mnist import input_data

from kl_estimators import naive_estimator as compute_kl
from mmd_utils import compute_mmd, compute_energy
from utils import get_data, generate_data, thinning_fn, sample_data


parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, default='test')
parser.add_argument('--weighted', default=False, action='store_true', dest='weighted',
    help='Chooses whether to use weighted MMD.')
parser.add_argument('--do_p', default=False, action='store_true', dest='do_p',
    help='Choose whether to use P, instead of TP')
parser.add_argument('--data_dim', type=int, default=2)
args = parser.parse_args()
tag = args.tag
weighted = args.weighted
do_p = args.do_p
data_dim = args.data_dim

data_num = 10000
latent_dim = 10

batch_size = 64  # MIW will split batch into 4 groups.
noise_dim = 10
h_dim = 10
learning_rate_init = 1e-4
log_iter = 1000
log_dir = 'results/mmd_{}'.format(tag)
max_iter = 25000


# Load data.
#(data_raw,
# data_raw_weights,
# data_raw_unthinned,
# data_raw_unthinned_weights,
# data_normed,
# data_raw_mean,
# data_raw_std) = generate_data(data_num, data_dim, latent_dim, with_latents=False, m_weight=2.)
(data_raw,
 data_raw_weights,
 data_raw_unthinned,
 data_raw_unthinned_weights,
 data_normed,
 data_raw_mean,
 data_raw_std) = get_data(data_dim, with_latents=False)


def sigmoid_cross_entropy_with_logits(logits, labels):
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, targets=labels)


def plot(generated, data_raw, data_raw_unthinned, it, mmd_gen_vs_unthinned):
    gen_v1 = generated[:, 0] 
    gen_v2 = generated[:, 1] 
    raw_v1 = [d[0] for d in data_raw]
    raw_v2 = [d[1] for d in data_raw]
    raw_unthinned_v1 = [d[0] for d in data_raw_unthinned]
    raw_unthinned_v2 = [d[1] for d in data_raw_unthinned]

    # Will use normalized data for evaluation of D.
    data_normed = to_normed(data_raw)

    # Evaluate D on grid.
    grid_gran = 20
    grid1 = np.linspace(min(data_raw[:, 0]), max(data_raw[:, 0]), grid_gran)

    fig = plt.figure()
    gs = GridSpec(8, 4)
    ax_joint = fig.add_subplot(gs[1:4, 0:3])
    ax_marg_x = fig.add_subplot(gs[0, 0:3], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1:4, 3], sharey=ax_joint)

    ax_joint.scatter(raw_v1, raw_v2, c='gray', alpha=0.3)
    ax_joint.scatter(gen_v1, gen_v2, alpha=0.3)
    ax_joint.set_aspect('auto')

    #ax_thinning = ax_joint.twinx()
    #ax_thinning.plot(grid1, thinning_fn(grid1, is_tf=False), color='red', alpha=0.3)

    ax_marg_x.hist([raw_v1, gen_v1], bins=30, color=['gray', 'blue'],
        label=['d', 'g'], alpha=0.3, normed=True)
    ax_marg_y.hist([raw_v2, gen_v2], bins=30, color=['gray', 'blue'],
        label=['d', 'g'], alpha=0.3, normed=True, orientation="horizontal",)
    ax_marg_x.legend()
    ax_marg_y.legend()

    # Turn off tick labels on marginals
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    ########
    # EVEN MORE PLOTTING.
    ax_raw = fig.add_subplot(gs[5:8, 0:3], sharex=ax_joint)
    ax_raw_marg_x = fig.add_subplot(gs[4, 0:3], sharex=ax_raw)
    ax_raw_marg_y = fig.add_subplot(gs[5:8, 3], sharey=ax_raw)
    ax_raw.scatter(raw_unthinned_v1, raw_unthinned_v2, c='gray', alpha=0.1)
    ax_raw_marg_x.hist(raw_unthinned_v1, bins=30, color='gray',
        label='d', alpha=0.3, normed=True)
    ax_raw_marg_y.hist(raw_unthinned_v2, bins=30, color='gray',
        label='d', orientation="horizontal", alpha=0.3, normed=True)
    plt.setp(ax_raw_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_raw_marg_y.get_yticklabels(), visible=False)
    ########

    plt.suptitle('mmdgan. it: {}, mmd_gen_vs_unthinned: {:.4f}'.format(
        it, mmd_gen_vs_unthinned))

    plt.savefig('{}/{}.png'.format(log_dir, it))
    plt.close()


def get_sample_z(m, n):
    return np.random.normal(0., 1., size=[m, n])


def upper(mat):
    return tf.matrix_band_part(mat, 0, -1) - tf.matrix_band_part(mat, 0, 0)


def to_raw(d, index=None):
    if index:
        return d * data_raw_std[index] + data_raw_mean[index]
    else:
        return d * data_raw_std + data_raw_mean


def to_normed(d, index=None):
    if index:
        return (d - data_raw_mean[index]) /  data_raw_std[index]
    else:
        return (d - data_raw_mean) /  data_raw_std


################################################################################
# BEGIN: Build model.
def dense(x, width, activation, batch_residual=False):
    if not batch_residual:
        x_ = layers.dense(x, width, activation=activation)
        return layers.batch_normalization(x_)
    else:
        x_ = layers.dense(x, width, activation=activation, use_bias=False)
        return layers.batch_normalization(x_) + x


def discriminator(inputs, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse) as d_vs:
        layer = dense(inputs, h_dim, activation=tf.nn.elu)
        layer = dense(layer, h_dim, activation=tf.nn.elu, batch_residual=True)
        layer = dense(layer, h_dim, activation=tf.nn.elu, batch_residual=True)
        d_logit = dense(layer, 1, activation=None)
        d_prob = tf.nn.sigmoid(d_logit)
    d_vars = tf.contrib.framework.get_variables(d_vs)
    return d_prob, d_logit, d_vars 


def generator(z, reuse=False):
    #inputs = tf.concat(axis=1, values=[z, x])
    with tf.variable_scope('generator', reuse=reuse) as g_vs:
        layer = dense(z, h_dim, activation=tf.nn.elu)
        layer = dense(layer, h_dim, activation=tf.nn.elu, batch_residual=True)
        #layer = dense(layer, h_dim, activation=tf.nn.elu, batch_residual=True)
        g = dense(layer, data_dim, activation=None)  # Outputing xy pairs.
    g_vars = tf.contrib.framework.get_variables(g_vs)
    return g, g_vars


def tf_median(v):
    m = v.get_shape()[0]//2
    return tf.nn.top_k(v, m).values[m-1]


def compute_mmd_iw_median_of_means(input1, input2, input1_weights):
    """Wrapper on compute_mmd_iw, to compute median of means.i
    
    Split input into groups, compute MMD on each, and do backprop on the median
    of those MMDs.
    """
    k1_in1, k2_in1, k3_in1, k4_in1 = tf.split(input1, 4)
    k1_in2, k2_in2, k3_in2, k4_in2 = tf.split(input2, 4)
    k1_in1_w, k2_in1_w, k3_in1_w, k4_in1_w = tf.split(input1_weights, 4)
    
    mmd1 = compute_mmd_iw(k1_in1, k1_in2, k1_in1_w)
    mmd2 = compute_mmd_iw(k2_in1, k2_in2, k2_in1_w)
    mmd3 = compute_mmd_iw(k3_in1, k3_in2, k3_in1_w)
    mmd4 = compute_mmd_iw(k4_in1, k4_in2, k4_in1_w)

    median_of_mmds = tf_median(tf.stack([mmd1, mmd2, mmd3, mmd4], axis=0))
    return median_of_mmds


def compute_mmd_iw(in1, in2, in1_weights):
    """Computes MMD between two batches of d-dimensional inputs.
    
    In this setting, in1 is real and in2 is generated, so in1 
    has weights.
    """
    batch_size = in1.shape[0]  # Does not use global var, since batch is split.

    num_combos_xx = tf.to_float(batch_size * (batch_size - 1) / 2)
    num_combos_yy = tf.to_float(batch_size * (batch_size - 1) / 2)

    v = tf.concat([in1, in2], 0)
    VVT = tf.matmul(v, tf.transpose(v))
    sqs = tf.reshape(tf.diag_part(VVT), [-1, 1])
    sqs_tiled_horiz = tf.tile(sqs, tf.transpose(sqs).get_shape())
    exp_object = sqs_tiled_horiz - 2 * VVT + tf.transpose(sqs_tiled_horiz)
    
    K = 0
    sigma_list = [0.01, 1.0, 2.0]
    for sigma in sigma_list:
        #gamma = 1.0 / (2.0 * sigma ** 2)
        K += tf.exp(-0.5 * (1 / sigma) * exp_object)
    K_xx = K[:batch_size, :batch_size]
    K_yy = K[batch_size:, batch_size:]
    K_xy = K[:batch_size, batch_size:]
    K_xx_upper = upper(K_xx)
    K_yy_upper = upper(K_yy)

    ######################
    # Originally written like this.
    #x_unnormed = v[:batch_size, :1] * data_raw_std[0] + data_raw_mean[0]
    #weights_x = 1. / thinning_fn(x_unnormed)
    #weights_x_tiled_horiz = tf.tile(weights_x, [1, batch_size])
    #p1_weights = weights_x_tiled_horiz
    #p2_weights = tf.transpose(p1_weights) 
    #p1p2_weights = p1_weights * p2_weights
    #p1p2_weights_upper = upper(p1p2_weights)
    #p1p2_weights_upper_normed = p1p2_weights_upper / tf.reduce_sum(p1p2_weights_upper)
    #p1_weights_normed = p1_weights / tf.reduce_sum(p1_weights)
    #Kw_xx_upper = K_xx * p1p2_weights_upper_normed
    #Kw_xy = K_xy * p1_weights_normed

    # Importance-weighted.
    weights_tiled_horiz = tf.tile(in1_weights, [1, batch_size])
    p1_weights = weights_tiled_horiz
    p2_weights = tf.transpose(p1_weights) 
    p1p2_weights = p1_weights * p2_weights
    p1p2_weights_upper = upper(p1p2_weights)
    Kw_xx_upper = K_xx * p1p2_weights_upper
    Kw_xy = K_xy * p1_weights

    mmd = (tf.reduce_sum(Kw_xx_upper) / num_combos_xx +
           tf.reduce_sum(K_yy_upper) / num_combos_yy -
           2 * tf.reduce_mean(Kw_xy))

    return mmd
        
###############################################################################
# Build model.
lr = tf.Variable(learning_rate_init, name='lr', trainable=False)
lr_update = tf.assign(lr, tf.maximum(lr * 0.5, 1e-8), name='lr_update')

z = tf.placeholder(tf.float32, shape=[batch_size, noise_dim], name='z')
z_sample = tf.placeholder(tf.float32, shape=[None, noise_dim], name='z_sample')
x = tf.placeholder(tf.float32, shape=[batch_size, data_dim], name='x')
x_weights = tf.placeholder(tf.float32, shape=[batch_size, 1], name='x_weights')

g, g_vars = generator(z, reuse=False)
g_sample, _ = generator(z_sample, reuse=True)
d_real, d_logit_real, d_vars = discriminator(x, reuse=False)
d_fake, d_logit_fake, _ = discriminator(g, reuse=True)

# Define losses.
mmd = compute_mmd_iw_median_of_means(x, g, x_weights)
g_loss = mmd

g_optim = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(
    g_loss, var_list=g_vars)

# End: Build model.
################################################################################


# Start session.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# train()
for it in range(max_iter):
    x_batch, x_batch_weights = sample_data(data_normed, data_raw_weights, batch_size)
    z_batch = get_sample_z(batch_size, noise_dim)

    _, g_loss_ = sess.run(
            [g_optim, g_loss],
        feed_dict={
            z: z_batch,
            x: x_batch,
            x_weights: x_batch_weights})

    if it % 100000 == 9999:
        sess.run(lr_update)

    if it % log_iter == 0:
        n_sample = 10000 
        z_sample_input = get_sample_z(n_sample, noise_dim)
        g_out = sess.run(g_sample, feed_dict={z_sample: z_sample_input})
        generated = np.array(g_out) * data_raw_std + data_raw_mean
        # Compute MMD between simulations and unthinned (target) data.
        mmd_gen_vs_unthinned, _ = compute_mmd(
            generated[np.random.choice(n_sample, 500)],
            data_raw_unthinned[np.random.choice(data_num, 500)])
        # Compute energy between simulations and unthinned (target) data.
        energy_gen_vs_unthinned = compute_energy(
            generated[np.random.choice(n_sample, 500)],
            data_raw_unthinned[np.random.choice(data_num, 500)])
        # Compute KL between simulations and unthinned (target) data.
        kl_gen_vs_unthinned = compute_kl(
            generated[np.random.choice(n_sample, 500)],
            data_raw_unthinned[np.random.choice(data_num, 500)], k=5)

        if data_dim == 2:
            fig = plot(generated, data_raw, data_raw_unthinned, it,
                mmd_gen_vs_unthinned)

        if np.isnan(g_loss_):
            sys.exit('got nan')
        
        # Print diagnostics.
        print("#################")
        lr_ = sess.run(lr)
        print('mmd_{}'.format(tag))
        print('Iter: {}, lr={:.4f}'.format(it, lr_))
        print('  median of mmds / g_loss: {:.4}'.format(g_loss_))
        print('  mmd_gen_vs_unthinned: {:.4}'.format(mmd_gen_vs_unthinned))
        print(data_raw[np.random.choice(data_num, 1), :5])
        print
        print(generated[:1, :5])
        with open(os.path.join(log_dir, 'scores_mmd.txt'), 'a') as f:
            f.write(str(mmd_gen_vs_unthinned)+'\n')
        with open(os.path.join(log_dir, 'scores_energy.txt'), 'a') as f:
            f.write(str(energy_gen_vs_unthinned)+'\n')
        with open(os.path.join(log_dir, 'scores_kl.txt'), 'a') as f:
            f.write(str(kl_gen_vs_unthinned)+'\n')
