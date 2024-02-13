def batched_index_select(values, indices, dim = 1):
    # https://github.com/lucidrains/egnn-pytorch/blob/main/egnn_pytorch/egnn_pytorch.py#L10
    # values [10 5 5 3]
    # indices[10 5 2]
    # dim 2

    value_dims = values.shape[(dim + 1):]  # [3]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices)) # [10 5 5 3], [10 5 2]
    indices = indices[(..., *((None,) * len(value_dims)))] # [10 5 2 1]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims) # [10 5 2 3]
    value_expand_len = len(indices_shape) - (dim + 1) # 1
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)
