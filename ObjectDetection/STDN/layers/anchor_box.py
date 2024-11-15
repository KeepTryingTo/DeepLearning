import torch
from math import sqrt


class AnchorBox(object):

    def __init__(self,
                 map_sizes,
                 aspect_ratios):
        super(AnchorBox, self).__init__()

        self.map_sizes = map_sizes  # [1, 3, 5, 9, 18, 36]
        self.aspect_ratios = aspect_ratios  # [1.6, 2, 3]

        self.scales = self.get_scales()

    def get_scales(self, scale_min=0.2, scale_max=1.05):
        scales = [0.1]

        num_map_sizes = len(self.map_sizes)

        scale_diff = scale_max - scale_min

        for k in range(1, num_map_sizes + 1):
            scale = scale_min + scale_diff / (num_map_sizes - 1) * (k - 1)
            scales.append(round(scale, 2))

        scales.reverse()
        return scales

    def get_boxes(self):
        boxes = []

        for k, map_size in enumerate(self.map_sizes):
            num_elements = map_size ** 2
            for i in range(num_elements):
                row = i // map_size
                col = i % map_size

                cx = (col + 0.5) / map_size
                cy = (row + 0.5) / map_size

                scale = self.scales[k + 1]
                scale_next = self.scales[k]
                scale_next = sqrt(scale * scale_next)

                boxes.append([cx, cy, scale, scale])
                boxes.append([cx, cy, scale_next, scale_next])

                for ratio in self.aspect_ratios:
                    ratio = sqrt(ratio)
                    boxes.append([cx, cy, scale * ratio, scale / ratio])
                    boxes.append([cx, cy, scale / ratio, scale * ratio])

        output = torch.Tensor(boxes).view(-1, 4)
        output.clamp_(max=1, min=0)
        return output.float()
