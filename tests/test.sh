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