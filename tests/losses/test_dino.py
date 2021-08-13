import torch
from solo.losses import DINOLoss


def test_dino_loss():
    b, f, num_epochs = 32, 128, 20
    p = torch.randn(b, f).requires_grad_()
    p_momentum = torch.randn(b, f)

    dino_loss = DINOLoss(
        num_prototypes=f,
        warmup_teacher_temp=0.4,
        teacher_temp=0.7,
        warmup_teacher_temp_epochs=10,
        student_temp=0.1,
        num_epochs=num_epochs,
    )

    loss = dino_loss(p, p_momentum)
    initial_loss = loss.item()
    assert loss != 0

    for i in range(20):
        dino_loss.epoch = i

        loss = dino_loss(p, p_momentum)
        loss.backward()
        p.data.add_(-0.5 * p.grad)

        p.grad = None

    assert loss < initial_loss
