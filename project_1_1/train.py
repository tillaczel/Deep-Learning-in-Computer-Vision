import os

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

wandb.init(project='p1', entity='dlcv')

@hydra.main(config_path='config', config_name="default")
def run_training(cfg : DictConfig):
    print(OmegaConf.to_yaml(cfg))

    cfg_file = os.path.join(wandb.run.dir, 'config.yaml')
    with open(cfg_file, 'w') as fh:
        fh.write(OmegaConf.to_yaml(cfg))

    train_dataloader, test_dataloader = get_data() #TODO
    model = None # TODO...

    callbacks = []
    callbacks.append(pl.callbacks.EarlyStopping(patience=10, monitor='val_loss'))
    callbacks.append(pl.callbacks.ModelCheckpoint(dirpath=wandb.run.dir,
                                                  monitor='val_loss',
                                                  filename='model',
                                                  verbose=True,
                                                  period=1))
    wandb.save('*.ckpt') # should keep it up to date
    # callbacks.append(WandBFilesync(filename='model.ckpt', period=10)) # not sure if this one is necessary

    logger = pl.loggers.WandbLogger()
    logger.watch(model)
    # TODO: we can use this to put configuration into nice table in wandb
    # logger.log_hyperparams(hyperparams)

    gpus = 0
    if torch.cuda.is_available():
        gpus = 1
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, default_root_dir="training/logs", max_epochs=cfg["max_epochs"], gpus=gpus)

    # trainer.tune(lit_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate

    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=test_dataloader)
    # trainer.test(model, datamodule=data_module)

    # TODO: visualizations
