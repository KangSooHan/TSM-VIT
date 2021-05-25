import torch.nn

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embedding = Embedding(config, img_size = img_size)


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes

        self.transformer = Transformer(config, img_size, vis)
        self.head = nn.Linear(config.d_model, num_classes)

    def forward(self, x, labels):
        x, attn_weights = self.transformer()
