import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class CNN_3(nn.Module):

    def __init__(self, num_classes=2):
        super(CNN_3, self).__init__()
        self.features = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0), # out : [batch_size, n_features_conv, height, width]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
    #     self.classifier = nn.Sequential(
    #         nn.Linear(32*5*5, 32),
    #         nn.ReLU(inplace=True),
    #         nn.Dropout(),
    #         nn.Linear(32, 2),
    #         # nn.ReLU(inplace=True),
    #         # nn.Dropout(),
    #     )
    #
    # def forward(self, x):
    #     x = self.features(x)
    #     print(x.size())
    #     # x = x.view(x.size()[0], -1)
    #     x = x.view(-1, 32*5*5)
    #     # print(x.size())
    #     x = self.classifier(x)
    #     return x

        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32 * 5 * 5, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    model.classifier[1] = nn.Linear(256 * 1 * 1, 4096)
    model.classifier[6] = nn.Linear(4096, 2)
    return model
