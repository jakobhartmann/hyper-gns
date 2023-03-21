import wandb

class Logger:
    def __init__(self, use_wandb, wandb_project = None, wandb_entity = None, wandb_resume = None, wandb_run_id = None, config = None):
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project = wandb_project, entity = wandb_entity, resume = wandb_resume, id = wandb_run_id, config = config)

    def log(self, data, step = None):
        if self.use_wandb:
            wandb.log(data, step)

    def config(self, config):
        if self.use_wandb:
            wandb.config(config)