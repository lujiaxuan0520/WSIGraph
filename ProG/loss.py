import torch
from torch import nn
from typing import Optional, List
import torch.nn.functional as F


class MultiLabelBCELoss(nn.Module):
    """Binary Cross Entropy loss over each label seperately, then averaged"""

    def __init__(self, weight=None) -> None:
        super().__init__()
        self.weight = weight
        self.bce = nn.BCEWithLogitsLoss(
            reduction="none" if weight is not None else "mean")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss of the logit and targets

        Args:
            logits (torch.Tensor): Logits for the slide with the shape: B x nr_classes
            targets (torch.Tensor): Targets one-hot encoded with the shape: B x nr_classes

        Returns:
            torch.Tensor: Slide loss
        """
        device = logits.device

        # make sure the two tensor are with the same size
        if logits.ndim < targets.ndim:
            logits = logits.unsqueeze(0)
        elif logits.ndim > targets.ndim:
            targets = targets.squeeze(1)
            # if targets.ndim > 1:
            #     targets = targets.squeeze(1)
            # else:
            #     targets = targets.squeeze()

        # while logits.ndim > targets.ndim:
        #     targets = targets.unsqueeze(-1)
        # while targets.ndim > logits.ndim:
        #     logits = logits.unsqueeze(0)

        # num_classes = logits.size(1)
        # targets = F.one_hot(targets, num_classes=num_classes).to(torch.float32)
        # print("logits", logits)
        # print("targets:", targets)
        # assert logits.shape == targets.shape, "Logits and targets must have the same shape"


        if self.weight is None:
            # return self.bce(input=logits, target=targets.to(torch.float32))
            return self.bce(input=logits, target=targets.to(device).to(torch.float32))
        else:
            # loss = self.bce(input=logits, target=targets.to(torch.float32))
            loss = self.bce(input=logits, target=targets.to(device).to(torch.float32))
            weighted_loss = loss * self.weight.to(loss.device)
            return weighted_loss.mean()


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        device = logits.device
        # targets = torch.argmax(targets, dim=1) # change one-hot to label

        if targets.dtype != torch.long:
            targets = targets.long()

        return self.loss(logits, targets.to(device))


def get_loss(name=None, device=None):
    # if name is not None:
    #     config = config[name]
    # # loss_class = dynamic_import_from("losses", config["class"])
    # loss_class = dynamic_import_from(
    #     "loss.common_loss", config["class"])
    # criterion = loss_class(**config.get("params", {}))
    if name == 'MultiLabelBCELoss':
        criterion = MultiLabelBCELoss()
    elif name == 'WeightedCrossEntropyLoss':
        criterion = WeightedCrossEntropyLoss()
    return criterion.to(device) if device is not None else criterion


class SlideOnlyCriterion(torch.nn.Module):
    def __init__(self, device, loss_name) -> None:
        super().__init__()
        self.slide_criterion = get_loss(loss_name, device)
        # self.instance_criterion = get_loss(loss, "instance", device)
        # self.instance_loss_weight = loss.get(
        #     "params", {}).get("instance_weight", 0.5)
        # assert (
        #     0.0 <= self.instance_loss_weight <= 1.0
        # ), f"instance weight loss must be between 0 and 1, but is {self.instance_loss_weight}"
        # self.slide_loss_weight = 1.0 - self.instance_loss_weight
        self.slide_loss_weight = 1.0
        self.device = device

    def forward(
            self,
            slide_logits: Optional[torch.Tensor] = None,
            slide_labels: Optional[torch.Tensor] = None,
            # instance_logits: Optional[torch.Tensor] = None,
            # instance_labels: Optional[torch.Tensor] = None,
            # instance_associations: Optional[List[int]] = None,
            drop_slide: Optional[bool] = False,
            # drop_instance: Optional[bool] = False,
    ):
        assert (
                slide_logits is not None and slide_labels is not None
        ), "Cannot use combined criterion without slide input"
        # assert (
        #     instance_logits is not None and instance_labels is not None
        # ), "Cannot use combined criterion without instance input"
        # instance_labels = instance_labels.to(self.device)
        slide_labels = slide_labels.to(self.device)

        slide_loss = self.slide_criterion(
            logits=slide_logits,
            targets=slide_labels
        )
        # instance_loss = self.instance_criterion(
        #     logits=instance_logits,
        #     targets=instance_labels,
        #     instance_associations=instance_associations,
        # )

        # if drop_slide:
        #     combined_loss = self.slide_loss_weight * 0 + \
        #         self.instance_loss_weight * instance_loss
        # elif drop_instance:
        #     combined_loss = self.slide_loss_weight * \
        #         slide_loss + self.instance_loss_weight * 0
        # else:
        #     combined_loss = (
        #         self.slide_loss_weight * slide_loss + self.instance_loss_weight * instance_loss
        #     )

        combined_loss = (
                self.slide_loss_weight * slide_loss
        )

        # print('only slide loss')
        # combined_loss = slide_loss
        # return combined_loss, slide_loss.detach().cpu(), instance_loss.detach().cpu()
        return combined_loss, slide_loss.detach().cpu()