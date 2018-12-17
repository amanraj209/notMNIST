import sys

sys.path.extend(['..'])

import tensorflow as tf

from data_loader.notMNIST import notMNISTDataLoaderNumpy
from models.notMNISTmodel import notMNISTModel
from trainers.notMNIST_trainer import notMNISTTrainer

from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import DefinedSummarizer
from utils.utils import get_args


def main():
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    create_dirs([config.summary_dir, config.checkpoint_dir])
    sess = tf.Session()

    data_loader = notMNISTDataLoaderNumpy(config)
    model = notMNISTModel(data_loader, config)
    logger = DefinedSummarizer(sess, summary_dir=config.summary_dir,
                               scalar_tags=['train/loss_per_epoch', 'train/acc_per_epoch',
                                            'test/loss_per_epoch','test/acc_per_epoch',
                                            'val/acc_per_epoch', 'val/loss_per_epoch'])

    trainer = notMNISTTrainer(sess, model, config, logger, data_loader)
    trainer.train()


if __name__ == '__main__':
    main()
