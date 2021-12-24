import torch
from typing import Tuple, Union, Optional, Dict
import kornia.augmentation as K
from kornia.augmentation.random_generator import bbox_generator
from torch.distributions import Uniform, Beta
from kornia.constants import Resample
from kornia.augmentation.utils.helpers import _adapted_rsampling, _adapted_uniform
from kornia.augmentation.utils.param_validation import _joint_range_check

def random_crop_generator(
    batch_size: int,
    input_size: Tuple[int, int],
    size: Union[Tuple[int, int], torch.Tensor],
    resize_to: Optional[Tuple[int, int]] = None,
    same_on_batch: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ```crop``` transformation for crop transform.

    Args:
        batch_size (int): the tensor batch size.
        input_size (tuple): Input image shape, like (h, w).
        size (tuple): Desired size of the crop operation, like (h, w).
            If tensor, it must be (B, 2).
        resize_to (tuple): Desired output size of the crop, like (h, w). If None, no resize will be performed.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.

    Example:
        >>> _ = torch.manual_seed(0)
        >>> crop_size = random_crop_size_generator(
        ...     3, (30, 30), scale=torch.tensor([.7, 1.3]), ratio=torch.tensor([.9, 1.]))['size']
        >>> crop_size
        tensor([[26, 29],
                [27, 28],
                [25, 28]], dtype=torch.int32)
        >>> random_crop_generator(3, (30, 30), size=crop_size, same_on_batch=False)
        {'src': tensor([[[ 1,  3],
                 [29,  3],
                 [29, 28],
                 [ 1, 28]],
        <BLANKLINE>
                [[ 2,  3],
                 [29,  3],
                 [29, 29],
                 [ 2, 29]],
        <BLANKLINE>
                [[ 0,  2],
                 [27,  2],
                 [27, 26],
                 [ 0, 26]]]), 'dst': tensor([[[ 0,  0],
                 [28,  0],
                 [28, 25],
                 [ 0, 25]],
        <BLANKLINE>
                [[ 0,  0],
                 [27,  0],
                 [27, 26],
                 [ 0, 26]],
        <BLANKLINE>
                [[ 0,  0],
                 [27,  0],
                 [27, 24],
                 [ 0, 24]]])}
    """
    if not isinstance(size, torch.Tensor):
        size = torch.tensor(size).repeat(batch_size, 1)
    assert size.shape == torch.Size([batch_size, 2]), \
        f"If `size` is a tensor, it must be shaped as (B, 2). Got {size.shape}."

    x_diff = input_size[1] - size[:, 1] + 1
    y_diff = input_size[0] - size[:, 0] + 1

    if (x_diff < 0).any() or (y_diff < 0).any():
        raise ValueError("input_size %s cannot be smaller than crop size %s in any dimension."
                         % (str(input_size), str(size)))

    if same_on_batch:
        # If same_on_batch, select the first then repeat.
        x_start = _adapted_uniform((batch_size,), 0, x_diff[0], same_on_batch).long()
        y_start = _adapted_uniform((batch_size,), 0, y_diff[0], same_on_batch).long()
    else:
        x_start = _adapted_uniform((1,), 0, x_diff, same_on_batch).long()
        y_start = _adapted_uniform((1,), 0, y_diff, same_on_batch).long()

    crop_src = bbox_generator(x_start.view(-1), y_start.view(-1), size[:, 1] - 1, size[:, 0] - 1)

    if resize_to is None:
        crop_dst = bbox_generator(
            torch.tensor([0] * batch_size), torch.tensor([0] * batch_size), size[:, 1] - 1, size[:, 0] - 1)
    else:
        crop_dst = torch.tensor([[
            [0, 0],
            [resize_to[1] - 1, 0],
            [resize_to[1] - 1, resize_to[0] - 1],
            [0, resize_to[0] - 1],
        ]]).repeat(batch_size, 1, 1)

    return dict(src=crop_src,
                dst=crop_dst)

def random_crop_size_generator(
        batch_size: int,
        size: Tuple[int, int],
        scale: torch.Tensor,
        ratio: torch.Tensor,
        power: int,
        same_on_batch: bool = False) -> Dict[str, torch.Tensor]:
    r"""Get cropping heights and widths for ```crop``` transformation for resized crop transform.

    Args:
        batch_size (int): the tensor batch size.
        size (Tuple[int, int]): expected output size of each edge.
        scale (tensor): range of size of the origin size cropped with (2,) shape.
        ratio (tensor): range of aspect ratio of the origin aspect ratio cropped with (2,) shape.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.

    Examples:
        >>> _ = torch.manual_seed(0)
        >>> random_crop_size_generator(3, (30, 30), scale=torch.tensor([.7, 1.3]), ratio=torch.tensor([.9, 1.]))
        {'size': tensor([[26, 29],
                [27, 28],
                [25, 28]], dtype=torch.int32)}
    """
    _joint_range_check(scale, "scale")
    _joint_range_check(ratio, "ratio")

    # 10 trails for each element
    area = _adapted_power((batch_size, 10), scale[0] * size[0] * size[1],
                          scale[1] * size[0] * size[1], power, same_on_batch)
    log_ratio = _adapted_power((batch_size, 10), torch.log(ratio[0]),
                               torch.log(ratio[1]), power, same_on_batch)
    aspect_ratio = torch.exp(log_ratio)

    w = torch.sqrt(area * aspect_ratio).int()
    h = torch.sqrt(area / aspect_ratio).int()
    # Element-wise w, h condition
    cond = ((0 < h) * (h < size[1]) * (0 < w) * (w < size[0])).int()
    cond_bool = torch.sum(cond, dim=1) > 0

    w_out = w[torch.arange(0, batch_size), torch.argmax(cond, dim=1)]
    h_out = h[torch.arange(0, batch_size), torch.argmax(cond, dim=1)]

    if not cond_bool.all():
        # Fallback to center crop
        in_ratio = float(size[0]) / float(size[1])
        if (in_ratio < min(ratio)):
            w_ct = torch.tensor(size[0])
            h_ct = torch.round(w_ct / min(ratio))
        elif (in_ratio > max(ratio)):
            h_ct = torch.tensor(size[1])
            w_ct = torch.round(h_ct * max(ratio))
        else:  # whole image
            w_ct = torch.tensor(size[0])
            h_ct = torch.tensor(size[1])
        w_ct = w_ct.int()
        h_ct = h_ct.int()

        w_out = w_out.where(cond_bool, w_ct)
        h_out = h_out.where(cond_bool, h_ct)

    return dict(size=torch.stack([w_out, h_out], dim=1))

def _adapted_power(shape: Union[Tuple, torch.Size],
                   low: Union[float, int, torch.Tensor],
                   high: Union[float, int, torch.Tensor],
                   power: int,
                   same_on_batch=False) -> torch.Tensor:
    r"""The uniform sampling function that accepts 'same_on_batch'.

    If same_on_batch is True, all values generated will be exactly same given a batch_size (shape[0]).
    By default, same_on_batch is set to False.
    """
    if not isinstance(low, torch.Tensor):
        low = torch.tensor(low, dtype=torch.float32)
    if not isinstance(high, torch.Tensor):
        high = torch.tensor(high, dtype=torch.float32)
    dist = Uniform(0, 1)
    sample = _adapted_rsampling(shape, dist, same_on_batch)
    sample = sample**power
    sample = low + sample * (high - low)
    return sample


class RandomResizedCropCustom(K.RandomResizedCrop):
    def __init__(
        self, size: Tuple[int, int], scale: Union[torch.Tensor, Tuple[float, float]] = (0.08, 1.0),
        ratio: Union[torch.Tensor, Tuple[float, float]] = (3. / 4., 4. / 3.),
        interpolation: Optional[Union[str, int, Resample]] = None,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        return_transform: bool = False, same_on_batch: bool = False,
        align_corners: bool = False, p: float = 1.,
        power: int = 1,
    ) -> None:
        super(RandomResizedCropCustom, self).__init__(size, scale, ratio, interpolation, resample, return_transform, align_corners)
        self.power = power
        
    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        target_size: torch.Tensor = random_crop_size_generator(
            batch_shape[0], self.size, self.scale, self.ratio, self.power, same_on_batch=self.same_on_batch)['size']
        return random_crop_generator(batch_shape[0], (batch_shape[-2], batch_shape[-1]), target_size,
                                        resize_to=self.size, same_on_batch=self.same_on_batch)