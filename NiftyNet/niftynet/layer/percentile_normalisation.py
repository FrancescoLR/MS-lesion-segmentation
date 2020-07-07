# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import numpy.ma as ma

from niftynet.layer.base_layer import Layer
from niftynet.layer.binary_masking import BinaryMaskingLayer


class PercentileNormalisationLayer(Layer):
    """
    This class defines image-level normalisation by subtracting
    the nth percentile value and dividing by the difference between the mth
    and nth percentile that are set as variable `cutoff` in the configuration.
    """

    def __init__(self, image_name, binary_masking_func=None, cutoff=[0.05, 0.95]):
        self.cutoff = cutoff
        self.image_name = image_name
        super(PercentileNormalisationLayer, self).__init__(name='perctl_norm')
        self.binary_masking_func = None
        if binary_masking_func is not None:
            assert isinstance(binary_masking_func, BinaryMaskingLayer)
            self.binary_masking_func = binary_masking_func

    def layer_op(self, image, is_training, mask=None):
        if isinstance(image, dict):
            image_data = np.asarray(image[self.image_name], dtype=np.float32)
        else:
            image_data = np.asarray(image, dtype=np.float32)

        if isinstance(mask, dict):
            image_mask = mask.get(self.image_name, None)
        elif mask is not None:
            image_mask = mask
        elif self.binary_masking_func is not None:
            image_mask = self.binary_masking_func(image_data)
        else:
            # no access to mask, default to the entire image
            image_mask = np.ones_like(image_data, dtype=np.bool)

        if image_data.ndim == 3:
            image_data = percentile_transformation(image_data, image_mask, self.cutoff, is_training)
        if image_data.ndim == 5:
            for m in range(image_data.shape[4]):
                for t in range(image_data.shape[3]):
                    image_data[..., t, m] = percentile_transformation(
                        image_data[..., t, m], image_mask[..., t, m], self.cutoff, is_training)

        if isinstance(image, dict):
            image[self.image_name] = image_data
            if isinstance(mask, dict):
                mask[self.image_name] = image_mask
            else:
                mask = {self.image_name: image_mask}
            return image, mask
        else:
            return image_data, image_mask


def percentile_transformation(image, mask, cutoff, is_training):
    # make sure image is a monomodal volume
    masked_img = ma.masked_array(image, np.logical_not(mask))
    prct = np.percentile(masked_img, 100. * np.array(cutoff))
    intensity_range = max(prct[1] - prct[0], 1e-6)
    image = (image - prct[0]) / intensity_range

    if is_training and self.net_param.percentile_normalisation:
        return image + np.reshape(dctmtx(intensity_range, 4, image), image.shape)
    else:
        return image

def dctmtx(intensity_range, dct_order, data):
    data = data.flatten()
    M = np.array([0, 0, 0, 0])
    S = np.array([[.12, -.1, 0, 0], [-.1, .12, 0, 0],
                  [0, 0, .5, -.5], [0, 0, -.5, .5]])/1000.*intensity_range
    coeffs = np.maximum(np.minimum(np.random.multivariate_normal(M,S),1),-1);
    data = data.flatten()
    C = np.zeros([data.shape[0], dct_order])
    C[:,0] = np.ones(data.shape[0])/np.sqrt(intensity_range)
    for k in range(1, dct_order):
        C[:,k] = np.sqrt(2/intensity_range)*np.cos(np.pi*(2.*data+1)*k/(2*intensity_range))
    return np.dot(C, coeffs)
