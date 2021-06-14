from omegaconf import DictConfig, OmegaConf

from project_2.src.utils import download_file
from project_2.src.engine import EngineModule


def run_eal(cfg: DictConfig):
    # load experiment config
    download_file(cfg.run_id, 'train_config.yaml')
    train_cfg = OmegaConf.load('train_config.yaml')

    cfg_file = os.path.join(wandb.run.dir, 'eval_config.yaml')
    with open(cfg_file, 'w') as fh:
        fh.write(OmegaConf.to_yaml(cfg))
    wandb.save(cfg_file)  # this will force sync it

    download_file(cfg.run_id, "model.ckpt")
    engine = EngineModule.load_from_checkpoint("model.ckpt", config=train_cfg)