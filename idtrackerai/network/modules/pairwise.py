import torch


def PairEnum(x, mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    if mask is not None:
        xmask = mask.view(-1, 1).repeat(1, x.size(1))
        # dim 0: #sample, dim 1:#feature
        x1 = x1[xmask].view(-1, x.size(1))
        x2 = x2[xmask].view(-1, x.size(1))
    return x1, x2


def Class2Simi(x, mode='cls', mask=None):
    """
    Give a 1d torch tensor with classes in dense format, returns the pairwise similarity matrix liniarized. A mask can
    be applied to discard some elements of the similarity matrix.

    :param x: 1d torch tensor with classes in dense format
    :param mode: 'cls' for classification 'hinge' for clustering
    :param mask: 2d torch tensor with the mask to be applied to the pairwise similarity matrix
    :return: 1d torch tensor with the elements to be considered
    """
    # Convert class label to pairwise similarity
    n = x.nelement()
    assert (n - x.ndimension() + 1) == n, 'Dimension of Label is not right'
    expand1 = x.view(-1, 1).expand(n, n)
    expand2 = x.view(1, -1).expand(n, n)
    out = expand1 - expand2
    out[out != 0] = -1  # dissimilar pair: label=-1
    out[out == 0] = 1  # Similar pair: label=1
    if mode == 'cls':
        out[out == -1] = 0  # dissimilar pair: label=0
    if mode == 'hinge':
        out = out.float()  # hingeloss require float type
    if mask is None:
        out = out.view(-1)
    else:
        mask = mask.detach()
        out = out[mask]
    return out
