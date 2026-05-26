from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import numpy as np
np.random.seed(10)

def save_accuracy(FLAGS, model, Accuracy_accountant, time, Budgets_accountant=None, nbytes1=None, nbytes2=None):

    save_dir = os.path.join(os.getcwd(), FLAGS.save_dir, 'res_{}'.format(FLAGS.version), FLAGS.dataset, FLAGS.model, ('noniid{}'.format(FLAGS.noniid_level) if FLAGS.noniid else 'iid'), (FLAGS.eps if FLAGS.dpsgd else 'nodp'))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = "{}{}{}{}{}{}{}{}{}".format(FLAGS.N,
                                    ('-fedavg' if FLAGS.fedavg else ''), 
                                    ('-wavg' if FLAGS.weiavg else ''), 
                                    ('-pro{}_{}'.format(FLAGS.proj_dims, FLAGS.lanczos_iter) if FLAGS.projection else ''),
                                    ('-wpro{}_{}'.format(FLAGS.proj_dims, FLAGS.lanczos_iter) if FLAGS.proj_wavg else ''),
                                    ('-plus' if FLAGS.delay else ''),
                                    '-{}-bs{}'.format(FLAGS.local_steps, FLAGS.client_batch_size), 
                                    ('-decaylr{}'.format(FLAGS.lr) if FLAGS.lr_decay else '-constlr{}'.format(FLAGS.lr)),
                                            ('-time{}'.format(time+1)))
                                    
    with open(os.path.join(save_dir, filename + 'accuracy.csv'), "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if FLAGS.dpsgd:
            writer.writerow(Budgets_accountant)
        if FLAGS.delay:
            writer.writerow(nbytes1)
            writer.writerow(nbytes2)
        writer.writerow(Accuracy_accountant)


def save_loss(FLAGS, model, Loss_accountant, time, Budgets_accountant=None, nbytes1=None, nbytes2=None):

    save_dir = os.path.join(os.getcwd(), FLAGS.save_dir, 'res_{}'.format(FLAGS.version), FLAGS.dataset, FLAGS.model,
                            ('noniid{}'.format(FLAGS.noniid_level) if FLAGS.noniid else 'iid'),
                            (FLAGS.eps if FLAGS.dpsgd else 'nodp'))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = "{}{}{}{}{}{}{}{}{}".format(FLAGS.N,
                                         ('-fedavg' if FLAGS.fedavg else ''),
                                         ('-wavg' if FLAGS.weiavg else ''),
                                         ('-pro{}_{}'.format(FLAGS.proj_dims,
                                                             FLAGS.lanczos_iter) if FLAGS.projection else ''),
                                         ('-wpro{}_{}'.format(FLAGS.proj_dims,
                                                              FLAGS.lanczos_iter) if FLAGS.proj_wavg else ''),
                                         ('-plus' if FLAGS.delay else ''),
                                         '-{}-bs{}'.format(FLAGS.local_steps, FLAGS.client_batch_size),
                                         ('-decaylr{}'.format(FLAGS.lr) if FLAGS.lr_decay else '-constlr{}'.format(
                                             FLAGS.lr)),
                                           ('-time{}'.format(time+1)))

    with open(os.path.join(save_dir, filename + 'loss.csv'), "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if FLAGS.dpsgd:
            writer.writerow(Budgets_accountant)
        if FLAGS.delay:
            writer.writerow(nbytes1)
            writer.writerow(nbytes2)
        writer.writerow(Loss_accountant)


def print_loss_and_accuracy(global_loss, accuracy, stage='validation'):
    print(' - Current Model has a loss of:           %s' % global_loss)
    print(' - The Accuracy on the ' + stage + ' set is: %s' % accuracy)
    print('--------------------------------------------------------------------------------------')
    print('--------------------------------------------------------------------------------------')


def print_new_comm_round(real_round):
    print('--------------------------------------------------------------------------------------')
    print('------------------------ Communication round %s ---------------------------------------' % str(real_round))
    print('--------------------------------------------------------------------------------------')


