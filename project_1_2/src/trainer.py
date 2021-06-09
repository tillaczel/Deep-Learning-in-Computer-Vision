import torch
import pytorch_lightning as pl
import wandb


def get_trainer(cfg, engine):
    callbacks = list()
    callbacks.append(pl.callbacks.EarlyStopping(patience=cfg.training.early_stopping.stopping_patience,
                                                monitor=cfg.training.early_stopping.monitor))
    callbacks.append(pl.callbacks.ModelCheckpoint(dirpath=wandb.run.dir,
                                                  monitor=cfg.training.model_checkpoint.monitor,
                                                  filename='model',
                                                  verbose=False,
                                                  period=1))
    callbacks.append(pl.callbacks.progress.ProgressBar())

    gpus = 0
    if torch.cuda.is_available():
        gpus = -1

    logger = pl.loggers.WandbLogger()
    logger.watch(engine)

    # TODO: we can use this to put configuration into nice table in wandb
    logger.log_hyperparams(cfg)

    trainer = pl.Trainer(callbacks=callbacks, logger=logger, default_root_dir="training/logs",
                         max_epochs=cfg.training.max_epochs, gpus=gpus)
    return trainer

def get_test_trainer(cfg, engine):
    logger = pl.loggers.WandbLogger()
    logger.watch(engine)
    gpus = 0
    if torch.cuda.is_available():
        gpus = -1
    logger.log_hyperparams({"run_id": cfg.run_id})
    return pl.Trainer(logger=logger, default_root_dir="eval/logs", gpus=gpus)