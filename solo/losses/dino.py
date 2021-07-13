import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np


class DINOLoss(nn.Module):
    def __init__(
        self,
        num_prototypes: int,
        warmup_teacher_temp: float,
        teacher_temp: float,
        warmup_teacher_temp_epochs: float,
        num_epochs: int,
        student_temp: float = 0.1,
        num_crops: int = 2,
        center_momentum: float = 0.9,
    ):
        """Auxiliary module to compute DINO's loss.

        Args:
            num_prototypes (int): number of prototypes.
            warmup_teacher_temp (float): base temperature for the temperature schedule
                of the teacher.
            teacher_temp (float): final temperature for the teacher.
            warmup_teacher_temp_epochs (float): number of epochs for the cosine annealing schedule.
            num_epochs (int): total number of epochs.
            student_temp (float, optional): temperature for the student. Defaults to 0.1.
            num_crops (int, optional): number of crops/views. Defaults to 2.
            center_momentum (float, optional): momentum for the EMA update of the center of
                mass of the teacher. Defaults to 0.9.
        """

        super().__init__()
        self.epoch = 0
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.num_crops = num_crops
        self.register_buffer("center", torch.zeros(1, num_prototypes))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
                np.ones(num_epochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor) -> torch.Tensor:
        """Computes DINO's loss given a batch of logits of the student and a batch of logits of the
        teacher.

        Args:
            student_output (torch.Tensor): NxP Tensor containing student logits for all views.
            teacher_output (torch.Tensor): NxP Tensor containing teacher logits for all views.

        Returns:
            torch.Tensor: DINO loss.
        """

        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.num_crops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[self.epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        """Updates the center for DINO's loss using exponential moving average.

        Args:
            teacher_output (torch.Tensor): NxP Tensor containing teacher logits of all views.
        """

        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()
        batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
