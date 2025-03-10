import logging
import os
import time

import torch
from click.core import batch
from torch.utils.tensorboard import SummaryWriter
from final_project.code.src.policy.dataset import DataTransformer

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(
        self,
        model,
        train_dataloader,
        dev_dataloader,
        criterion,
        optimizer,
        device=None,
        checkpoint_dir="checkpoints",
        checkpoint_name="best_checkpoint.pt",
        experiment_name="experiment",
        log_dir="runs",
        early_stopping_patience=10,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )
        self.model.to(self.device)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.log_dir = log_dir
        self.writer = SummaryWriter(f"{log_dir}/{experiment_name}")
        self.early_stopping_patience = (
            early_stopping_patience  # how many eval cycles to wait if no improvement
        )

        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def train(self, num_epochs, eval_interval=1):
        best_accuracy = 0.0
        early_stopping_counter = 0

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            start_time = time.time()

            # Training loop
            for i, (inputs, labels) in enumerate(self.train_dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_dataloader)
            self.writer.add_scalar("Training Loss", avg_loss, epoch)

            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, "
                f"Time: {time.time() - start_time:.2f}s"
            )

            # Evaluate on dev set every eval_interval epochs
            if (epoch + 1) % eval_interval == 0:
                train_accuracy = self.evaluate(self.train_dataloader)
                dev_accuracy = self.evaluate(self.dev_dataloader)
                self.writer.add_scalar("Train Accuracy", train_accuracy, epoch)
                self.writer.add_scalar("Dev Accuracy", dev_accuracy, epoch)

                # Save checkpoint if we have the best performance on the dev set
                if best_accuracy < dev_accuracy:
                    best_accuracy = dev_accuracy
                    self.save_checkpoint(epoch, self.optimizer)
                    logger.info(f"New best model saved at epoch {epoch+1}.")
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
            # add early stopping condition
            if early_stopping_counter > self.early_stopping_patience:
                logger.info("Early stopping activated.")
                break

        self.writer.close()

    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                predictions = self.model.predict(inputs)
                total += labels.size(0)  # add number of samples in the batch
                # all correct predictions
                correct += (predictions == labels).sum().item()

        accuracy = correct / total
        return accuracy

    def save_checkpoint(self, epoch, optimizer):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Best checkpoint saved to {checkpoint_path}")
        return

    def load_checkpoint(self, checkpoint_path, optimizer=None):
        checkpoint = torch.load(checkpoint_path)
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if optimizer:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]
            logger.info(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}")
        except KeyError:
            logger.warning("Checkpoint is not a dictionary of metadata.")
            self.model.load_state_dict(checkpoint)
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return

    def sample_action(self, state, deterministic=True):
        if type(state) != torch.Tensor:
            state = torch.tensor(state.__array__(), dtype=torch.float32)

        state = state.to(self.device)
        if state.ndim == 3:
            state = state.unsqueeze(0)  # Now state shape becomes (1, 3, h, w)

        if state.ndim == 3:
            batch = True
        else:
            batch = False

        if deterministic:
            action = self.model.predict(state, batch=batch)
        else:
            action = self.model.predict_stochastic(state)
        return action
