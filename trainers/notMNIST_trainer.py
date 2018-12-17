from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np

import tensorflow as tf

from utils.metrics import AverageMeter
from utils.logger import DefinedSummarizer


class notMNISTTrainer(BaseTrain):
    def __init__(self, sess, model, config, logger, data_loader):
        super(notMNISTTrainer, self).__init__(sess, model, config, logger, data_loader)
        self.model.load(self.sess)
        self.summarizer = logger
        self.x, self.y, self.is_training = tf.get_collection('inputs')
        self.train_op, self.loss_node, self.acc_node = tf.get_collection('train')

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch(cur_epoch)
            self.sess.run(self.model.increment_cur_epoch_tensor)
            self.test(cur_epoch)
            self.valid(cur_epoch)

    def valid(self, epoch=None):
        self.data_loader.initialize(self.sess, is_train=False, is_val=True)

        tt = tqdm(range(self.data_loader.num_iterations_val), total=self.data_loader.num_iterations_val,
                  desc="epoch-{}-".format(epoch))

        loss_per_epoch = AverageMeter()
        acc_per_epoch = AverageMeter()

        for cur_it in tt:
            loss, acc = self.sess.run([self.loss_node, self.acc_node],
                                      feed_dict={self.is_training: False})
            loss_per_epoch.update(loss)
            acc_per_epoch.update(acc)

        summaries_dict = {'val/loss_per_epoch': loss_per_epoch.val,
                          'val/acc_per_epoch': acc_per_epoch.val}
        self.summarizer.summarize(self.model.global_step_tensor.eval(self.sess), summaries_dict)

        print("""Validation -> Val-{}  loss:{:.4f} -- acc:{:.4f}""".format(epoch, loss_per_epoch.val,
                                                                                 acc_per_epoch.val))
        tt.close()


    def train_epoch(self, epoch=None):
        self.data_loader.initialize(self.sess, is_train=True, is_val=False)

        tt = tqdm(range(self.data_loader.num_iterations_train), total=self.data_loader.num_iterations_train,
                  desc="epoch-{}-".format(epoch))

        loss_per_epoch = AverageMeter()
        acc_per_epoch = AverageMeter()


        for cur_it in tt:
            loss, acc = self.train_step()
            loss_per_epoch.update(loss)
            acc_per_epoch.update(acc)

        self.sess.run(self.model.global_epoch_inc)

        summaries_dict = {'train/loss_per_epoch': loss_per_epoch.val,
                          'train/acc_per_epoch': acc_per_epoch.val}
        self.summarizer.summarize(self.model.global_step_tensor.eval(self.sess), summaries_dict)

        self.model.save(self.sess)

        print("""Training -> Epoch-{}  loss:{:.4f} -- acc:{:.4f}""".format(epoch, loss_per_epoch.val, acc_per_epoch.val))

        tt.close()

    def train_step(self):
        _, loss, acc = self.sess.run([self.train_op, self.loss_node, self.acc_node],
                                     feed_dict={self.is_training: True})
        return loss, acc

    def test(self, epoch):
        self.data_loader.initialize(self.sess, is_train=False, is_val = False)

        tt = tqdm(range(self.data_loader.num_iterations_test), total=self.data_loader.num_iterations_test,
                  desc="Val-{}-".format(epoch))

        loss_per_epoch = AverageMeter()
        acc_per_epoch = AverageMeter()

        for cur_it in tt:
            loss, acc = self.sess.run([self.loss_node, self.acc_node],
                                      feed_dict={self.is_training: False})
            loss_per_epoch.update(loss)
            acc_per_epoch.update(acc)

        summaries_dict = {'test/loss_per_epoch': loss_per_epoch.val,
                          'test/acc_per_epoch': acc_per_epoch.val}
        self.summarizer.summarize(self.model.global_step_tensor.eval(self.sess), summaries_dict)

        print("""Testing -> Val-{}  loss:{:.4f} -- acc:{:.4f}""".format(epoch, loss_per_epoch.val, acc_per_epoch.val))

        tt.close()
