import math

import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "needle": NeedleSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None: # for curriculum learning, truncate dimension of x
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b


class NeedleSampler(DataSampler):
    """
    NeedleSampler generates data for the needle in haystack task.
    Here, each sample is a context of n_points elements (each element is 1-dimensional)
    so that the overall shape is [batch, n_points, 1].
    """
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        # It is advisable that for the needle task you set n_dims = 1.
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        # Generate a tensor of shape [b_size, n_points, self.n_dims].
        # For needle in haystack, self.n_dims is typically 1.
        
        
        # context_value = torch.randint(0, 10) # Set the 'haystack' value # integer value might just let the model memorize?
        context_value = torch.randn(1,)
              
        
        if seeds is None:
            # xs = torch.randn(b_size, n_points, self.n_dims)
            xs = torch.ones(b_size, n_points, self.n_dims)
            xs = xs * context_value
        else:
            NotImplementedError
        # else:
        #     xs = torch.zeros(b_size, n_points, self.n_dims)
        #     generator = torch.Generator()
        #     assert len(seeds) == b_size
        #     for i, seed in enumerate(seeds):
        #         generator.manual_seed(seed)
        #         xs[i] = torch.randn(n_points, self.n_dims, generator=generator)
        
        
        # Insert needle
        needle_index = torch.randint(0, n_points, (b_size, 1))
        needle_strength = torch.randn(1,)
        
        for i in range(b_size):
            xs[i, needle_index[i], :] += needle_strength
            
        return xs, needle_index