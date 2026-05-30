#!/usr/bin/env bash

set -euo pipefail

policy_name=wallx
task_name=${1}
task_config=${2}
seed=${3}
gpu_id=${4}
wallx_server_uri=${5:-ws://127.0.0.1:8000}
wallx_dataset_names=${6:-x2_normal}
wallx_exec_horizon=${7:-4}
wallx_action_scale=${8:-1.0}
wallx_action_dim=${9:-14}
wallx_state_dim=${10:-14}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo "[wallx-eval] server=${wallx_server_uri} dataset=${wallx_dataset_names}"
echo "[wallx-eval] action_dim=${wallx_action_dim} state_dim=${wallx_state_dim} exec_horizon=${wallx_exec_horizon}"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/wall-x/deploy_policy.yml \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting wallx-zeroshot \
    --seed "${seed}" \
    --policy_name "${policy_name}" \
    --wallx_server_uri "${wallx_server_uri}" \
    --wallx_dataset_names "${wallx_dataset_names}" \
    --wallx_exec_horizon "${wallx_exec_horizon}" \
    --wallx_action_scale "${wallx_action_scale}" \
    --wallx_action_dim "${wallx_action_dim}" \
    --wallx_state_dim "${wallx_state_dim}"
