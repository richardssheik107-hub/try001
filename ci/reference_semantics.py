import math


def sparse_softmax_index(src, index, eps=1e-16):
    out = [0.0] * len(src)
    groups = {}
    for i, group in enumerate(index):
        groups.setdefault(group, []).append(i)
    for members in groups.values():
        group_max = max(src[i] for i in members)
        exp_values = [math.exp(src[i] - group_max) for i in members]
        denom = sum(exp_values) + eps
        for i, e in zip(members, exp_values):
            out[i] = e / denom
    return out


def sparse_softmax_ptr(src, ptr, eps=1e-16):
    out = [0.0] * len(src)
    for group in range(len(ptr) - 1):
        start, end = ptr[group], ptr[group + 1]
        if end <= start:
            continue
        group_max = max(src[start:end])
        exp_values = [math.exp(v - group_max) for v in src[start:end]]
        denom = sum(exp_values) + eps
        for offset, e in enumerate(exp_values, start):
            out[offset] = e / denom
    return out


def assert_close(actual, expected, tol=1e-7):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert abs(a - e) <= tol, (actual, expected)


def main():
    assert_close(
        sparse_softmax_index([1.0, 1.0, 1.0, 1.0], [0, 0, 1, 2]),
        [0.5, 0.5, 1.0, 1.0],
    )
    assert_close(
        sparse_softmax_ptr([1.0, 1.0, 1.0, 1.0], [0, 2, 3, 4]),
        [0.5, 0.5, 1.0, 1.0],
    )

    values = [10000.0, 10001.0, 10002.0]
    result = sparse_softmax_index(values, [7, 7, 7])
    assert all(math.isfinite(v) for v in result)
    assert abs(sum(result) - 1.0) <= 1e-7

    values = [0.5, 0.3, 0.8, 0.2, 0.6]
    result = sparse_softmax_index(values, [0, 0, 1, 1, 1])
    assert abs(sum(result[:2]) - 1.0) <= 1e-7
    assert abs(sum(result[2:]) - 1.0) <= 1e-7

    # Empty CSR group between [0:2] and [2:4].
    result = sparse_softmax_ptr([1.0, 2.0, 3.0, 4.0], [0, 2, 2, 4])
    assert abs(sum(result[:2]) - 1.0) <= 1e-7
    assert abs(sum(result[2:]) - 1.0) <= 1e-7

    print("reference SparseSoftmax semantics: PASS")


if __name__ == "__main__":
    main()
