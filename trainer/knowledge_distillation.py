import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader


def kd_loss(model_output, teacher_output, label, alpha=0.1, T=3):
    """
    Use Kullback-Leibler divergence for distillation loss

    :param model_output: The output tensor of the student model
    :param teacher_output: The output tensor of the teacher model
    :param label: The ground truth labels
    :param alpha: Weight between hard label loss and soft label loss
    :param T: Temperature for softening probability distribution
    :return: Knowledge distillation loss
    """
    KD_loss = nn.KLDivLoss()(F.log_softmax(model_output/T, dim=1), F.softmax(teacher_output/T, dim=1)) * (alpha*T*T) + F.cross_entropy(model_output, label) * (1. - alpha)

    return KD_loss


def image_trainkd(model, teacher_model, optimizer, loss_fn, training_data, test_data, num_epochs, scheduler=None):
    """
    Train a image classification model using knowledge distillation

    :param model: a pytorch Module which is the student model
    :param teacher_model: a pytorch Module which is the teacher model
    :param optimizer: a pytorch Optimizer for optimization
    :param loss_fn: a loss function that computes the distillation loss
    :param training_data: a pytorch Dataset for training
    :param test_data: a pytorch Dataset for testing
    :param num_epochs: the total number of epochs
    :param scheduler: a pytorch StepLR class for learning rate scheduling
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device).train()
    teacher_model.to(device).eval()

    train_loader = DataLoader(training_data)
    test_loader = DataLoader(test_data)

    for epoch in range(num_epochs):
        model.train()
        if scheduler is not None:
            scheduler.step()
        loader = tqdm(train_loader)

        train_acc = 0
        test_acc = 0
        for img, label in loader:
            output = model(img.to(device))
            teacher_output = teacher_model(img.to(device))
            preds = output.data.max(1)[1]
            corr = preds.eq(label.to(device).data).sum().item()
            train_acc += corr

            optimizer.zero_grad()
            loss = loss_fn(output, teacher_output, label=label.to(device))
            loss.backward()
            optimizer.step()

            loader.set_description(f"training... loss:{loss.item()}, right{train_acc}/{len(training_data)}")

        model.eval()

        loader = tqdm(test_loader)
        for img, label in loader:
            output = model(img.to(device))
            preds = output.data.max(1)[1]
            corr = preds.eq(label.to(device).data).sum().item()
            test_acc += corr

            loader.set_description(f"evaluating... right{test_acc}/{len(test_data)}")

        print(f"epoch:{epoch}/{num_epochs} training_acc:{train_acc/len(training_data)} test_acc:{test_acc/len(test_data)}")
