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
import time

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
parser.add_argument('--max_step', type=int, default=25000)
parser.add_argument('--log_step', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--estimator', type=str, default='sn', choices=['sn', 'iw'])

args = parser.parse_args()
tag = args.tag
weighted = args.weighted
do_p = args.do_p
data_dim = args.data_dim
max_step = args.max_step
log_step = args.log_step
batch_size = args.batch_size
learning_rate_init = args.learning_rate
estimator = args.estimator

latent_dim = 10
noise_dim = 10
h_dim = 10
log_dir = 'results/mmd_{}'.format(tag)


# Load data.
#(data_raw,
# data_raw_weights,
# data_raw_unthinned,
# data_raw_unthinned_weights,
# data_normed,
# data_raw_mean,
# data_raw_std) = generate_data(data_num, data_dim, latent_dim, with_latents=False, m_weight=2.)
(m_weight,
 data_raw,
 data_raw_weights,
 data_raw_unthinned,
 data_raw_unthinned_weights,
 data_normed,
 data_raw_mean,
 data_raw_std) = get_data(data_dim, with_latents=False)

data_num_original = data_raw.shape[0]
data_num = data_raw.shape[0]


## Make upsampled set.
## Given the thinned set, for each point, compute its weight, round to
## nearest integer "k", and add k-1 repetitions, making k of that value.
#data_raw_upsampled = []
#data_raw_upsampled_weights = []
#for i, val in enumerate(data_raw):
#    weight = data_raw_weights[i]
#    k = int(round(weight))
#    for _ in range(k - 1):
#        data_raw_upsampled.append(val)
#        data_raw_upsampled_weights.append(weight)
#data_raw_upsampled = np.reshape(data_raw_upsampled, [-1, data_dim])
#data_raw_upsampled_weights = np.reshape(data_raw_upsampled_weights, [-1, 1])
#
## Add upsamples to original data set.
#data_raw = np.concatenate((data_raw, data_raw_upsampled))
#data_raw_weights = np.concatenate(
#    (data_raw_weights, data_raw_upsampled_weights))
#assert data_raw.shape[0] == data_raw_weights.shape[0], \
#    'data and weights don\'t match'
#data_num = data_raw.shape[0]
#
## Shuffle data and weights.
#permutation_order = np.random.permutation(data_num)
#data_raw = data_raw[permutation_order]
#data_raw_weights = data_raw_weights[permutation_order]
#
## Compute normed version of data.
#data_normed = (data_raw - data_raw_mean) / data_raw_std


def sigmoid_cross_entropy_with_logits(logits, labels):
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, targets=labels)


def plot(generated, data_raw, data_raw_unthinned, log_dir, tag, step, measure_to_plot):
    gen_v1 = generated[:, 0] 
    gen_v2 = generated[:, 1] 
    raw_v1 = [d[0] for d in data_raw]
    raw_v2 = [d[1] for d in data_raw]
    raw_unthinned_v1 = [d[0] for d in data_raw_unthinned]
    raw_unthinned_v2 = [d[1] for d in data_raw_unthinned]

    # Evaluate D on grid.
    #grid_gran = 20
    #grid_x = np.linspace(min(data_raw[:, 0]), max(data_raw[:, 0]), grid_gran)
    #grid_y = np.linspace(min(data_raw[:, 1]), max(data_raw[:, 1]), grid_gran)
    #vals_on_grid = np.zeros((grid_gran, grid_gran))
    #for i in range(grid_gran):
    #    for j in range(grid_gran):
    #        grid_x_normed = (grid_x[i] - data_raw_mean[0]) / data_raw_std[0]
    #        grid_y_normed = (grid_y[j] - data_raw_mean[0]) / data_raw_std[0]
    #        vals_on_grid[i][j] = run_discrim([grid_x_normed, grid_y_normed])

    fig = plt.figure()
    gs = GridSpec(8, 4)
    ax_joint = fig.add_subplot(gs[1:4, 0:3])
    ax_marg_x = fig.add_subplot(gs[0, 0:3], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1:4, 3], sharey=ax_joint)

    ax_joint.scatter(raw_v1, raw_v2, c='gray', alpha=0.1)
    ax_joint.scatter(gen_v1, gen_v2, alpha=0.3)
    ax_joint.set_aspect('auto')
    #ax_joint.imshow(vals_on_grid, interpolation='nearest', origin='lower',
    #    alpha=0.3, aspect='auto',
    #    extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])

    bins = np.arange(-3, 3, 0.2)
    ax_marg_x.hist([raw_v1, gen_v1], bins=bins, color=['gray', 'blue'],
        label=['data', 'gen'], alpha=0.3, normed=True)
    ax_marg_y.hist([raw_v2, gen_v2], bins=bins, color=['gray', 'blue'],
        label=['data', 'gen'], orientation="horizontal", alpha=0.3, normed=True)
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
    ax_raw.scatter(raw_unthinned_v1, raw_unthinned_v2, c='green', alpha=0.1)
    ax_raw_marg_x.hist([raw_unthinned_v1, gen_v1], bins=bins, color=['green', 'blue'],
        label=['unthinned', 'gen'], alpha=0.3, normed=True)
    ax_raw_marg_y.hist([raw_unthinned_v2, gen_v2], bins=bins, color=['green', 'blue'],
        label=['unthinned', 'gen'], alpha=0.3, normed=True, orientation='horizontal')
    ax_raw_marg_x.legend()
    ax_raw_marg_y.legend()
    plt.setp(ax_raw_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_raw_marg_y.get_yticklabels(), visible=False)
    ########

    plt.suptitle('{}. step: {}, discrepancy: {:.4f}'.format(
        log_dir[8:], step, measure_to_plot))
    plt.savefig('{}/{}.png'.format(log_dir, step))
    plt.close()


def get_sample_z(m, n):
    return np.random.normal(0., 1., size=[m, n])


def upper(mat):
    return tf.matrix_band_part(mat, 0, -1) - tf.matrix_band_part(mat, 0, 0)


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


def compute_mmd_weighted(input1, input2, input1_weights, estimator):
    """Computes MMD between two batches of d-dimensional inputs.
    
    In this setting, input1 is real and input2 is generated, so input1 
    has weights.
    """
    num_combos_xx = tf.to_float(batch_size * (batch_size - 1) / 2)
    num_combos_yy = tf.to_float(batch_size * (batch_size - 1) / 2)

    v = tf.concat([input1, input2], 0)
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

    if estimator == 'iw':
        # Importance-weighted.
        weights_tiled_horiz = tf.tile(input1_weights, [1, batch_size])
        p1_weights = weights_tiled_horiz
        p2_weights = tf.transpose(p1_weights) 
        p1p2_weights = p1_weights * p2_weights
        p1p2_weights_upper = upper(p1p2_weights)
        Kw_xx_upper = K_xx * p1p2_weights_upper
        Kw_xy = K_xy * p1_weights

        mmd = (tf.reduce_sum(Kw_xx_upper) / num_combos_xx +
               tf.reduce_sum(K_yy_upper) / num_combos_yy -
               2 * tf.reduce_mean(Kw_xy))

    elif estimator == 'sn':
        # Self-normalized weights.
        weights_tiled_horiz = tf.tile(input1_weights, [1, batch_size])
        p1_weights = weights_tiled_horiz
        p2_weights = tf.transpose(p1_weights) 
        p1p2_weights = p1_weights * p2_weights
        p1p2_weights_upper = upper(p1p2_weights)
        p1p2_weights_upper_normed = p1p2_weights_upper / tf.reduce_sum(p1p2_weights_upper)
        p1_weights_normed = p1_weights / tf.reduce_sum(p1_weights)
        Kw_xx_upper = K_xx * p1p2_weights_upper_normed
        Kw_xy = K_xy * p1_weights_normed

        mmd = (tf.reduce_sum(Kw_xx_upper) +
               tf.reduce_sum(K_yy_upper) / num_combos_yy -
               2 * tf.reduce_sum(Kw_xy))

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
mmd = compute_mmd_weighted(x, g, x_weights, estimator)
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
# Start clock to time execution in chunks of log_step steps.
t0 = time.time()
for step in range(max_step):
    #x_batch, x_batch_weights = sample_data(data_normed, data_raw_weights, batch_size)
    # Do upsampling according to weights, then downsample to batch size.
    x_batch_preup, w_batch_preup = sample_data(data_normed, data_raw_weights, batch_size)
    x_batch = []
    w_batch = []
    for x_, w_ in zip(x_batch_preup, w_batch_preup):
        k = int(round(w_))
        for _ in range(k - 1):
            x_batch.append(x_)
            w_batch.append(w_)
    x_batch = np.reshape(x_batch, [-1, data_dim])
    w_batch = np.reshape(w_batch, [-1, 1])
    random_choice = np.random.choice(len(x_batch), batch_size)
    x_batch = x_batch[random_choice]
    w_batch = w_batch[random_choice]
    z_batch = get_sample_z(batch_size, noise_dim)

    _, g_loss_ = sess.run(
            [g_optim, g_loss],
        feed_dict={
            z: z_batch,
            x: x_batch,
            x_weights: w_batch})

    # TODO: Check whether lr_update helps here.
    if step % 100000 == 9999:
        sess.run(lr_update)

    if step > 0 and step % log_step == 0:
        # Stop clock after log_step training steps.
        t1 = time.time()
        chunk_time = t1 - t0

        n_sample = 1000 
        z_sample_input = get_sample_z(n_sample, noise_dim)
        g_out = sess.run(g_sample, feed_dict={z_sample: z_sample_input})
        generated = np.array(g_out) * data_raw_std + data_raw_mean
        # Compute MMD between simulations and unthinned (target) data.
        mmd_gen_vs_unthinned, _ = compute_mmd(
            generated[np.random.choice(n_sample, 1000)],
            data_raw_unthinned[np.random.choice(data_num_original, 1000)])
        # Compute energy between simulations and unthinned (target) data.
        energy_gen_vs_unthinned = compute_energy(
            generated[np.random.choice(n_sample, 1000)],
            data_raw_unthinned[np.random.choice(data_num_original, 1000)])
        # Compute KL between simulations and unthinned (target) data.
        kl_gen_vs_unthinned = compute_kl(
            generated[np.random.choice(n_sample, 1000)],
            data_raw_unthinned[np.random.choice(data_num_original, 1000)], k=5)

        if data_dim == 2:
            measure_to_plot = energy_gen_vs_unthinned
            fig = plot(generated, data_raw, data_raw_unthinned, log_dir, tag, step,
                measure_to_plot)

        if np.isnan(g_loss_):
            sys.exit('got nan')
        
        # Print diagnostics.
        print("#################")
        lr_ = sess.run(lr)
        print('mmd_{}'.format(tag))
        print('Iter: {}, lr={:.4f}'.format(step, lr_))
        print('  g_loss: {:.4}'.format(g_loss_))
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

        # Plot timing and performance together.
        with open(os.path.join(log_dir, 'perf.txt'), 'a') as f:
            model_type = 'mmd'
            model_subtype, model_dim, model_runnum = tag.split('_')
            model_dim = model_dim[3:]  # Drop "dim" from "dim*".
            model_runnum = model_runnum[3:]  # Drop "run" from "run*".
            f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(
                model_type, model_subtype, model_dim, model_runnum, step,
                g_loss_, mmd_gen_vs_unthinned, energy_gen_vs_unthinned,
                kl_gen_vs_unthinned, chunk_time))


        # Restart clock for next log_step training steps.
        t0 = time.time()
