  srcs_version = "PY3",
    deps = [
        # pip: absl:app
        "//sonnet",
        # pip: tensorflow
        # pip: tensorflow_datasets
    ],
)

snt_py_library(
name = "simple_mnist_library",
    srcs = ["simple_mnist.py"],
    deps = [
        # pip: absl:app
        "//sonnet",
        # pip: tensorflow
        # pip: tensorflow_datasets
    ],
)

snt_py_test(
    name = "simple_mnist_test",
    srcs = ["simple_mnist.py"],
    deps = [
        # pip: absl:app
        "//sonnet",
        # pip: tensorflow
        # pip: tensorflow_datasets
    ],
)

py_binary(
 name = "functional_mlp_mnist",
    srcs = ["functional_mnist.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl:app
        # pip: absl/logging
        "//sonnet",
        # pip: tensorflow
        # pip: tensorflow_datasets
    ],
)