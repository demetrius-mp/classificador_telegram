from typing import Any

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models.mobilenetv2 import mobilenet_v2


class Classifier:
    def __init__(self) -> None:
        self.transforms = Classifier.load_transformer()
        self.model = Classifier.get_cached_model()
        self.labels = sorted(
            [
                "abelha",
                "cachorro",
                "gato",
                "leao",
                "ovelha",
                "tartaruga",
                "vaca",
                "aguia",
                "cavalo",
                "girafa",
                "macaco",
                "porco",
                "tigre",
                "zebra",
                "aranha",
                "elefante",
                "jacare",
                "onca",
                "raposa",
                "tubarao",
            ]
        )

    @staticmethod
    def load_transformer():
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        return transform

    @staticmethod
    def get_cached_model():
        cache = torch.load("model.pth")

        model = mobilenet_v2()
        model.classifier[1] = torch.nn.Linear(1280, 20)
        model.load_state_dict(cache["model"])

        return model

    def __call__(self, img: Image.Image) -> str:
        self.model.eval()
        data = self.transforms(img)
        data: Any = data.unsqueeze(0)

        device = torch.device("cpu")

        with torch.no_grad():
            data = data.to(device)

            pred = self.model(data)
            res = pred.argmax(dim=1).cpu().tolist()[0]

            label = self.labels[res]

        return label


classifier = Classifier()
