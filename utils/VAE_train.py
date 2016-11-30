
import os
import math
import sys
import tensorflow as tf
import numpy as np
from VAE_input import openDataset
from itertools import izip
import time

def run_epoch(session, model, trainOp, dataset, configuration, mode='learn'):
    avg_cost = 0.0
    avg_partial = None
    batch_size = configuration.batch_size

    total_batch = int(dataset.num_examples / batch_size)

    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = dataset.next_batch(batch_size)
        if mode == 'learn':
            cost, partial, _ = session.run([model.cost, model.partial_costs, trainOp], feed_dict=model.createFeed(batch_xs, batch_ys))
        elif mode == 'evaluate':
            cost, partial, _ = session.run([model.cost_eval, model.partial_costs_eval, trainOp], feed_dict=model.createFeed(batch_xs, batch_ys))
        else:
            raise NotImplemented

        # Compute average loss
        avg_cost += cost / dataset.num_examples * batch_size

        if avg_partial is None:
            avg_partial = []
            for p in partial:
                avg_partial.append(p / dataset.num_examples * batch_size)
        else:
            for n,p in enumerate(partial):
                avg_partial[n] += p / dataset.num_examples * batch_size

    return avg_cost, avg_partial

def _add_eval_op(eval_name):
    eval_variable = tf.Variable(0, name=eval_name, trainable=False, dtype=tf.float32)
    tf.scalar_summary(eval_name, eval_variable)
    return eval_variable

def assignVariables(session, model, epoch, configuration):

    # assign beta for warm up
    if 'beta_warm_up' in model.customVariables:
        assert( configuration.warm_up >= 0.0 )
        if configuration.warm_up > 0:
            beta_warm_up = min(1.0, float(epoch) / float(configuration.warm_up))
            session.run(model.customVariables['beta_warm_up'].assign(beta_warm_up))
        if configuration.warm_up == 0:
            session.run(model.customVariables['beta_warm_up'].assign(1.0))

    #assign step for annealing learning rate
    step = float(epoch)
    session.run(model.customVariables['step'].assign(step))

def train(configuration, path, ModelClass):

    dataset = openDataset(datasetName=configuration.datasetName, small=False)
    dataset_train = dataset.train
    dataset_val = dataset.validation
    dataset_test = dataset.test

    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope('model', reuse=None):
            print 'Creating model'
            m = ModelClass(configuration)

        with tf.name_scope('evaluation'):
            loss_train_var = _add_eval_op('loss/01_train')
            loss_val_var = _add_eval_op('loss/02_val')
            loss_test_var = _add_eval_op('loss/03_test')

            partials_train_vars = []
            partials_val_vars = []
            partials_test_vars = []

            for partial in m.partials:
                partials_train_vars.append(_add_eval_op(partial+'/01_train'))
                partials_val_vars.append(_add_eval_op(partial+'/02_val'))
                partials_test_vars.append(_add_eval_op(partial+'/03_test'))

        with tf.name_scope('saver'):
            saver = tf.train.Saver(max_to_keep=(configuration.early_stopping+2))

        tf.initialize_all_variables().run()

        if configuration.classifier == True:
            best_val = 0
            best_cost_val = 0
            best_test = 0
        else:
            best_val = sys.float_info.max
            best_cost_val = sys.float_info.max
            best_test = sys.float_info.max
        best_partials_test = []
        best_partials_val = []

        without_improvement = 0

        total_time=0.

        print 'Training starts'
        for epoch in range(configuration.max_epoch):

            assignVariables(session, m, epoch, configuration)

            start_time = time.time()
            run_epoch(session, m, m.optimizer, dataset_train, configuration)
            end_time = time.time() - start_time

            total_time = total_time + end_time
            costs_train, partials_train = run_epoch(session, m, tf.no_op(), dataset_train, configuration, mode='learn')
            costs_val, partials_val = run_epoch(session, m, tf.no_op(), dataset_val, configuration, mode='evaluate')
            costs_test, partials_test = run_epoch(session, m, tf.no_op(), dataset_test, configuration, mode='evaluate')

            if configuration.classifier == True:
                cost_val_early_stopping = -1.*partials_val[0]
            else:
                cost_val_early_stopping = costs_val

            best_str = ''

            if epoch > configuration.min_epoch:
                if cost_val_early_stopping < best_val:
                    best_val = cost_val_early_stopping
                    best_cost_val = costs_val
                    best_partials_val = partials_val
                    best_test = costs_test
                    best_partials_test = partials_test
                    without_improvement = 0
                    best_str = 'best'
                    saver.save(session, path)
                else:
                    if configuration.early_stopping > 0 and without_improvement == configuration.early_stopping:
                        best_str = 'now'
                        break
                    else:
                        without_improvement += 1
                        best_str = str(without_improvement)

            print \
                '\n=Epoch:', '%04d=\n' % (epoch+1), \
                'train cost=', '{:.4f}'.format(costs_train), \
                '[', [ '{:.4f}'.format(mm) for mm in partials_train], ']\n', \
                'val cost=', "{:.4f}".format(costs_val), \
                '[', [ '{:.4f}'.format(mm) for mm in partials_val], ']\n', \
                'test cost=', "{:.4f}".format(costs_test), \
                '[', [ '{:.4f}'.format(mm) for mm in partials_test], ']\n', \
                'early-stopping: ' + best_str +'/' + str(configuration.early_stopping), \
                ', time elapsed {:.2f}'.format(end_time)

            session.run([loss_train_var.assign(costs_train), loss_val_var.assign(costs_val), loss_test_var.assign(costs_test)])

            for (num, (partial_train, partial_val, partial_test)) in enumerate(izip(partials_train, partials_val, partials_test)):
                session.run([partials_train_vars[num].assign(partial_train)])
                session.run([partials_val_vars[num].assign(partial_val)])
                session.run([partials_test_vars[num].assign(partial_test)])

            if math.isnan(costs_train) or math.isnan(costs_val) or math.isnan(costs_test):
                print 'nan'
                break

        print '\nTraining ends'

        print\
            '\n ===FINAL RESULTS===\n'\
            'test cost:', "{:.4f}".format(best_test), \
            '[', [ '{:.4f}'.format(mm) for mm in best_partials_test], ']\n', \
            'epochs: ', '%d\n' % (epoch+1)

        final_result = '_||_val_cost_' + str( best_cost_val ) + '_' + str( [ mm for mm in best_partials_val ] ) + '_||' + '_test_cost_' + str( best_test ) + '_' + str( [ mm for mm in best_partials_test ] ) +'_||_epochs_' + str(epoch+1) + '_||_time_' + str(total_time / (epoch*1.+1.))

    return final_result
