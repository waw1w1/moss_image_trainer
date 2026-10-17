"""Epoch boundaries shared by training, inspection and checkpoint validation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingPlan:
    batches_per_epoch: int
    updates_per_epoch: int
    max_steps: int
    max_epochs: int
    save_every_epochs: int
    save_every_steps: int

    def should_save(self, step, next_epoch, next_batch):
        if step == self.max_steps:
            return True
        if self.save_every_epochs:
            return next_batch == 0 and next_epoch % self.save_every_epochs == 0
        return step % self.save_every_steps == 0


def training_plan(training, batch_count):
    if type(batch_count) is not int or batch_count < 1:
        raise ValueError("Training requires at least one bucket batch per epoch")
    updates = (batch_count + training.gradient_accumulation_steps - 1) // training.gradient_accumulation_steps
    max_steps = training.max_epochs * updates if training.max_epochs else training.max_steps
    max_epochs = training.max_epochs or (max_steps + updates - 1) // updates
    return TrainingPlan(batch_count, updates, max_steps, max_epochs,
                        training.save_every_epochs, training.save_every)


def checkpoint_name(config, step, next_epoch, next_batch):
    if config.training.max_epochs:
        if next_batch != 0 or next_epoch < 1:
            raise ValueError("Epoch checkpoints can only be saved after a complete epoch")
        return f"epoch-{next_epoch:06d}"
    return f"step-{step:08d}"
