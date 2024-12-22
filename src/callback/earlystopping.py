from pathlib import Path
from loguru import logger
import torch
from huggingface_hub import create_branch, create_repo


class EarlyStopping:
    def __init__(
        self, config, patience=7, min_delta=0, mode="min", save_path="checkpoints"
    ):
        self.patience = config.early_stopping_patience or patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.config = config
        self.best_score = None
        self.early_stop = False
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.best_path = self.save_path / "best_model.pth"
        self.min_delta *= 1 if mode == "min" else -1

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        else:
            if self.mode == "min":
                improvement = self.best_score - score > self.min_delta
            else:
                improvement = score - self.best_score > self.min_delta

            if improvement:
                self.best_score = score
                self.counter = 0
                self.save_checkpoint(model)
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, model):
        logger.info("Saving best model")
        torch.save(model.state_dict(), self.best_path)
        try:
            create_repo(repo_id=f"pradanaadn/{self.config.repo_name}", exist_ok=True)

            create_branch(
                repo_id=f"pradanaadn/{self.config.repo_name}",
                repo_type="model",
                branch=self.config.architecture,
                exist_ok=True,
            )
            model.push_to_hub(
                repo_id=f"pradanaadn/{self.config.repo_name}",
                commit_message=f"Save best model checkpoint with F1-Score Macro {self.best_score}",
                branch=self.config.architecture,
                
            )
            logger.success(
                f"Success Saving best model with F1-Score Macro {self.best_score} "
            )
        except Exception as e:
            logger.error(f"Error pushing to hub: {str(e)}")
