import torch
import pytorch_lightning as pl


def get_trainer(cfg, engine):
    callbacks = list()
    callbacks.append(pl.callbacks.EarlyStopping(patience=cfg.training.early_stopping.stopping_patience,
                                                monitor=cfg.training.early_stopping.monitor))
    callbacks.append(pl.callbacks.ModelCheckpoint(dirpath=wandb.run.dir,
                                                  monitor=cfg.training.model_checkpoint.monitor,
                                                  filename='model',
                                                  verbose=True,
                                                  period=1))

    gpus = 0
    if torch.cuda.is_available():
        gpus = -1

    logger = pl.loggers.WandbLogger()
    logger.watch(engine)

    # TODO: we can use this to put configuration into nice table in wandb
    # logger.log_hyperparams(hyperparams)

    trainer = pl.Trainer(callbacks=callbacks, logger=logger, default_root_dir="training/logs",
                         max_epochs=cfg.training.max_epochs, gpus=gpus)
    return trainer