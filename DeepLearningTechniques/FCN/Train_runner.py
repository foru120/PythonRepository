from Dataloader import TrainDataLoader
from Train import Trainer


if __name__ == '__main__':
    loader = TrainDataLoader(n_validation=10, batch_size=100, img_size=256, data_path='/home/bjh/data')
    trainer = Trainer(data_loader=loader, img_size=256, n_class=2, batch_size=100, n_epoch=20, learning_rate=0.001, drop_out_rate=0.01, decay_rate=0.96, model_save_path='/home/bjh/new/2ds/model1/models', act_func='ce')
    trainer.train_()