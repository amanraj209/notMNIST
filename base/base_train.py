import tensorflow as tf


class BaseTrain:
    def __init__(self, sess, model, config, logger, data_loader=None):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        if data_loader is not None:
            self.data_loader = data_loader
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self, epoch=None):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError
