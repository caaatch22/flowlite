#!/bin/bash

test_forward() {
    echo "Running python3 -m pytest -v -k 'forward'"
    python3 -m pytest -v -k "forward"
}

test_backward() {
    echo "Running python3 -m pytest -v -k 'backward'"
    python3 -m pytest -l -v -k "backward"
}

test_topo_sort() {
    echo "Running python3 -m pytest -k 'topo_sort'"
    python3 -m pytest -k "topo_sort"
}

test_compute_gradient() {
    echo "Running python3 -m pytest -k 'compute_gradient'"
    python3 -m pytest -k "compute_gradient"
}

test_softmax_loss_pt() {
    echo "Running python3 -m pytest -k 'softmax_loss_pt'"
    python3 -m pytest -k "softmax_loss_pt"
}

test_nn_epoch_pt() {
    echo "Running python3 -m pytest -l -k 'nn_epoch_pt'"
    python3 -m pytest -l -k "nn_epoch_pt"
}


test_init() {
    echo "Running python3 -m pytest -v -k 'test_init'"
    python3 -m pytest -v -k "test_init"
}

test_nn_linear() {
    echo "Running python3 -m pytest -v -k "test_nn_linear""
    python3 -m pytest -v -k "test_nn_linear"
}

test_nn_relu() {
    echo "Running python3 -m pytest -v -k "test_nn_linear""
    python3 -m pytest -v -k "test_nn_linear"
}

test_nn_sequential() {
    echo "Running python3 -m pytest -v -k "test_nn_sequential""
    python3 -m pytest -v -k "test_nn_sequential"
}

test_op_logsumexp() {
    echo "Running python3 -m pytest -v -k "test_op_logsumexp""
    python3 -m pytest -v -k "test_op_logsumexp"
}

# TODO: change name to crossentropyloss in tests
test_nn_softmax_loss() {
    echo "Running python3 -m pytest -v -k "test_nn_softmax_loss""
    python3 -m pytest -v -k "test_nn_softmax_loss"
}

test_nn_flatten() {
    echo "Running python3 -m pytest -v -k "test_nn_flatten""
    python3 -m pytest -v -k "test_nn_flatten"
}

test_nn_batchnorm() {
    echo "Running python3 -m pytest -v -k "test_nn_batchnorm""
    python3 -m pytest -v -k "test_nn_batchnorm"
}

test_nn_dropout() {
    echo "Running python3 -m pytest -v -k "test_nn_dropout""
    python3 -m pytest -v -k "test_nn_dropout"
}

test_nn_residual() {
    echo "Running python3 -m pytest -v -k "test_nn_residual""
    python3 -m pytest -v -k "test_nn_residual"
}

if [ $# -eq 0 ]; then
    # if no args, run all function starts with test_
    for func in $(declare -F | awk '{print $3}' | grep '^test_'); do
        $func
    done
else
    # run functions test_$arg
    for arg in "$@"; do
        func="test_$arg"
        if declare -F | grep -q " $func$"; then
            $func
        else
            echo "$func not found"
        fi
    done
fi