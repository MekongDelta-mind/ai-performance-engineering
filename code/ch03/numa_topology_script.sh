#!/bin/bash

# NUMA topology binding script
# Binds only when the platform exposes an authoritative GPU NUMA node.

mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)

for GPU in "${GPUS[@]}"; do
    PCI_BUS=$(nvidia-smi -i "$GPU" --query-gpu=pci.bus_id --format=csv,noheader | head -1 | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')
    SYSFS_BUS="$PCI_BUS"
    if [[ "$SYSFS_BUS" == 00000000:* ]]; then
        SYSFS_BUS="${SYSFS_BUS:4}"
    fi

    NODE=""
    NUMA_PATH="/sys/bus/pci/devices/${SYSFS_BUS}/numa_node"
    if [[ -r "$NUMA_PATH" ]]; then
        NODE=$(cat "$NUMA_PATH")
    fi

    if [[ "$NODE" =~ ^[0-9]+$ ]]; then
        numactl --cpunodebind="$NODE" --membind="$NODE" \
                bash -c "CUDA_VISIBLE_DEVICES=$GPU python train.py --gpu $GPU" &
    else
        echo "GPU $GPU NUMA node unavailable; running without numactl binding" >&2
        bash -c "CUDA_VISIBLE_DEVICES=$GPU python train.py --gpu $GPU" &
    fi
done

# Wait for all background processes to complete
wait
