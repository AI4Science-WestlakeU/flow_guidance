import torch
import torch.nn.functional as F
from functools import partial
import numpy as np

def gaussian_2d_kernel(sigma, size):
    """Generate a 2D Gaussian kernel."""
    x = torch.arange(-size // 2 + 1., size // 2 + 1.)
    y = torch.arange(-size // 2 + 1., size // 2 + 1.)
    x, y = torch.meshgrid(x, y)
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

def gaussian_blur(x, sigma_blur, size_kernel):
    '''Blur a tensor image with Gaussian filter

    x: tensor image, NxCxWxH
    sigma: standard deviation of the Gaussian kernel
    '''
    kernel = gaussian_2d_kernel(sigma_blur, size_kernel).type_as(x)
    # uniform kernel
    kernel = kernel.view(1, 1, size_kernel, size_kernel)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)
    # kernel = kernel.flip(-1).flip(-2)
    return F.conv2d(x, kernel, stride=1, padding='same', groups=x.shape[1])


def half_mask(x):
    """
    Mask on x corresponding to replacing with zeros the up half of the image
    """
    d = x.shape[2] // 2

    mask = torch.ones_like(x)
    mask[:, :, :d, :] = 0
    return mask * x


def square_mask(x, half_size_mask):
    """
    Black square mask of 20 x 20 pixels at the center of the image
    """
    d = x.shape[2] // 2

    mask = torch.ones_like(x)
    mask[:, :, d - half_size_mask:d + half_size_mask,
         d - half_size_mask:d + half_size_mask] = 0
    return mask * x

def upsample(x, sf):
    '''s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros(
        (x.shape[0],
         x.shape[1],
         x.shape[2] *
         sf,
         x.shape[3] *
         sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf):
    '''s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]

# def downsample(x: torch.Tensor, sf: int = 2) -> torch.Tensor:
#     """
#     Perform block average downsampling on the input x.
#     :param x: Input image with shape (N, C, H, W)
#     :param sf: Scaling factor (sampling step)
#     :return: Downsampled result with shape (N, C, H//sf, W//sf)
#     """
#     N, C, H, W = x.shape
#     assert H % sf == 0, f"Input height {H} is not divisible by sf={sf}"
#     assert W % sf == 0, f"Input width {W} is not divisible by sf={sf}"
    
#     # reshape -> average within blocks -> get (N, C, H//sf, W//sf)
#     x = x.reshape(N, C, H // sf, sf, W // sf, sf)      # (N, C, H//sf, sf, W//sf, sf)
#     x_down = x.mean(dim=(3, 5))                        # average within blocks
#     return x_down


# def upsample(x: torch.Tensor, sf: int = 2) -> torch.Tensor:
#     """
#     Perform replication upsampling on the input x.
#     :param x: Input image with shape (N, C, H, W)
#     :param sf: Scaling factor
#     :return: Upsampled result with shape (N, C, H*sf, W*sf)
#     """
#     # repeat along H and W dimensions sf times
#     x_up = x.repeat(1, 1, sf, sf)
#     return x_up

def Mvp(A, vec):
    return A @ vec

def _safe_normalize(x, threshold=None):
    norm = torch.norm(x)
    if threshold is None:
        threshold = torch.finfo(norm.dtype).eps
    normalized_x = x / norm if norm > threshold else torch.zeros_like(x)
    return normalized_x, norm
def arnoldi(vec,    # Matrix vector product
            V,      # List of existing basis
            H,      # H matrix
            j):     # number of basis
    '''
    Arnoldi iteration to find the j th l2-orthonormal vector
    compute the j-1 th column of Hessenberg matrix
    '''
    _check_nan(vec, 'Matrix vector product is Nan')

    for i in range(j):
        H[i, j - 1] = torch.dot(vec, V[i])
        vec = vec - H[i, j-1] * V[i]
    new_v, vnorm = _safe_normalize(vec)
    H[j, j - 1] = vnorm
    return new_v


def _check_nan(vec, msg):
    if torch.isnan(vec).any():
        raise ValueError(msg)
    
def cal_rotation(a, b):
    '''
    Args:
        a: element h in position j
        b: element h in position j+1
    Returns:
        cosine = a / \sqrt{a^2 + b^2}
        sine = - b / \sqrt{a^2 + b^2}
    '''
    c = torch.sqrt(a * a + b * b)
    return a / c, - b / c
def apply_given_rotation(H, cs, ss, j):
    '''
    Apply givens rotation to H columns
    :param H:
    :param cs:
    :param ss:
    :param j:
    :return:
    '''
    # apply previous rotation to the 0->j-1 columns
    for i in range(j):
        tmp = cs[i] * H[i, j] - ss[i] * H[i + 1, j]
        H[i + 1, j] = cs[i] * H[i+1, j] + ss[i] * H[i, j]
        H[i, j] = tmp
    cs[j], ss[j] = cal_rotation(H[j, j], H[j + 1, j])
    H[j, j] = cs[j] * H[j, j] - ss[j] * H[j + 1, j]
    H[j + 1, j] = 0
    return H, cs, ss

def GMRES(A,                # Linear operator, matrix or function
          b,                # RHS of the linear system in which the first half has the same shape as grad_gx, the second half has the same shape as grad_fy
          x0=None,          # initial guess, tuple has the same shape as b
          max_iter=None,    # maximum number of GMRES iterations
          tol=1e-6,         # relative tolerance
          atol=1e-6,        # absolute tolerance
          track=False):     # If True, track the residual error of each iteration
    '''
    Return:
        sol: solution
        (j, err_history):
            j is the number of iterations used to achieve the target accuracy;
            err_history is a list of relative residual error at each iteration if track=True, empty list otherwise.
    '''
    if isinstance(A, torch.Tensor):
        Avp = partial(Mvp, A)
    elif hasattr(A, '__call__'):
        Avp = A
    else:
        raise ValueError('A must be a function or matrix')

    bnorm = torch.norm(b)

    if max_iter == 0 or bnorm < 1e-8:
        return b

    if max_iter is None:
        max_iter = b.shape[0]

    if x0 is None:
        x0 = torch.zeros_like(b)
        r0 = b
    else:
        r0 = b - Avp(x0)

    new_v, rnorm = _safe_normalize(r0)
    # initial guess residual
    beta = torch.zeros(max_iter + 1, device=b.device)
    beta[0] = rnorm
    err_history = []
    if track:
        err_history.append((rnorm / bnorm).item())

    V = []
    V.append(new_v)
    H = torch.zeros((max_iter + 1, max_iter + 1), device=b.device)
    cs = torch.zeros(max_iter, device=b.device)  # cosine values at each step
    ss = torch.zeros(max_iter, device=b.device)  # sine values at each step

    for j in range(max_iter):
        p = Avp(V[j])
        # Arnoldi iteration to get the j+1 th basis
        new_v = arnoldi(p, V, H, j + 1)
        V.append(new_v)

        H, cs, ss = apply_given_rotation(H, cs, ss, j)
        _check_nan(cs, f'{j}-th cosine contains NaN')
        _check_nan(ss, f'{j}-th sine contains NaN')
        beta[j + 1] = ss[j] * beta[j]
        beta[j] = cs[j] * beta[j]
        residual = torch.abs(beta[j + 1])
        if track:
            err_history.append((residual / bnorm).item())
        if residual < tol * bnorm or residual < atol:
            break
    y = torch.linalg.solve_triangular(
        H[0:j + 1, 0:j + 1], beta[0:j + 1].unsqueeze(-1), upper=True)  # j x j
    V = torch.stack(V[:-1], dim=0)
    sol = x0 + V.T @ y.squeeze(-1)
    return sol, (j, err_history)


# Function to create the downsampling matrix
def create_downsampling_matrix(H, W, sf, device):
    assert H % sf == 0 and W % sf == 0, "Image dimensions must be divisible by sf"

    H_ds, W_ds = H // sf, W // sf  # Downsampled dimensions
    N = H * W  # Total number of pixels in the original image
    M = H_ds * W_ds  # Total number of pixels in the downsampled image

    # Initialize downsampling matrix of size (M, N)
    downsample_matrix = torch.zeros((M, N), device=device)

    # Fill the matrix with 1s at positions corresponding to downsampling
    for i in range(H_ds):
        for j in range(W_ds):
            # The index in the downsampled matrix
            downsampled_idx = i * W_ds + j

            # The corresponding index in the original flattened matrix
            original_idx = (i * sf * W) + (j * sf)

            # Set the value to 1 to perform downsampling
            downsample_matrix[downsampled_idx, original_idx] = 1

    return downsample_matrix

# Adapted from deepinv
def bicubic_filter(factor=2):
    r"""
    Bicubic filter.

    It has size (4*factor, 4*factor) and is defined as

    .. math::

        \begin{equation*}
            w(x, y) = \begin{cases}
                (a + 2)|x|^3 - (a + 3)|x|^2 + 1 & \text{if } |x| \leq 1 \\
                a|x|^3 - 5a|x|^2 + 8a|x| - 4a & \text{if } 1 < |x| < 2 \\
                0 & \text{otherwise}
            \end{cases}
        \end{equation*}

    for :math:`x, y \in {-2\text{factor} + 0.5, -2\text{factor} + 0.5 + 1/\text{factor}, \ldots, 2\text{factor} - 0.5}`.

    :param int factor: downsampling factor
    """
    x = np.arange(start=-2 * factor + 0.5, stop=2 * factor, step=1) / factor
    a = -0.5
    x = np.abs(x)
    w = ((a + 2) * np.power(x, 3) - (a + 3) * np.power(x, 2) + 1) * (x <= 1)
    w += (
        (a * np.power(x, 3) - 5 * a * np.power(x, 2) + 8 * a * x - 4 * a)
        * (x > 1)
        * (x < 2)
    )
    w = np.outer(w, w)
    w = w / np.sum(w)
    return torch.Tensor(w).unsqueeze(0).unsqueeze(0)

def random_mask(x, p, seed=None):
    """
    Random mask on x
    """
    np.random.seed(42)
    mask = torch.from_numpy(np.random.binomial(n=1, p=1-p, size=(
        x.shape[0], x.shape[2], x.shape[3]))).to(x.device)

    return mask.unsqueeze(1) * x
