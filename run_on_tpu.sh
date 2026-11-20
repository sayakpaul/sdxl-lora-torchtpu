#!/usr/bin/env bash
# Run from the workstation with gcloud access. TPU_NAME selects an existing VM.
set -euo pipefail
cd "$(dirname "$0")"
preview_from=
if (( $# )); then
    if [[ $# != 2 || $1 != --preview-from || ! -s $2/pytorch_lora_weights.safetensors ]]; then
        echo "Usage: $0 [--preview-from RESULT_DIRECTORY]" >&2
        exit 2
    fi
    preview_from=$2
fi
project=${PROJECT:-$(gcloud config get-value project)}
zone=${ZONE:-us-east5-a}
job="sdxl-lora-$(date -u +%Y%m%d-%H%M%S)-$RANDOM"
vm=${TPU_NAME:-$job}
location=(--project="$project" --zone="$zone")
local_dir="runs/$job"
remote_dir="$job"
created=false
keep_vm=false

cleanup() {
    if [[ $created == true ]]; then
        if [[ $keep_vm == true ]]; then
            echo "VM retained for result recovery. Delete it after copying the run:" >&2
            echo "gcloud compute tpus tpu-vm delete $vm --project=$project --zone=$zone" >&2
        else
            gcloud compute tpus tpu-vm delete "$vm" "${location[@]}" --quiet
        fi
    fi
}
trap cleanup EXIT
mkdir -p runs
mkdir "$local_dir"
if [[ -z ${TPU_NAME:-} ]]; then
    gcloud compute tpus tpu-vm create "$vm" "${location[@]}" \
        --accelerator-type=v6e-1 --version=v2-alpha-tpuv6e
    created=true
fi
gcloud compute tpus tpu-vm ssh "$vm" "${location[@]}" --command="mkdir $remote_dir"
gcloud compute tpus tpu-vm scp run_sdxl_lora.py sdxl-torchtpu.patch \
    "$vm:$remote_dir/" "${location[@]}"

# Fresh Python processes give inference the chip's full memory after training.
remote_script='
    set -eu
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null || python3 -m pip install --user uv
'
if [[ -n $preview_from ]]; then
    gcloud compute tpus tpu-vm scp --recurse "$preview_from" "$vm:$remote_dir/result" "${location[@]}"
else
    remote_script+='
        mkdir result
        uv run --python 3.12 run_sdxl_lora.py --output result
    '
fi
remote_script+='
    uv run --python 3.12 run_sdxl_lora.py --output result --preview-only
'
status=0
gcloud compute tpus tpu-vm ssh "$vm" "${location[@]}" \
    --command="cd $remote_dir && $remote_script" 2>&1 | tee "$local_dir/console.log" || status=$?
if ! gcloud compute tpus tpu-vm scp --recurse "$vm:$remote_dir/result" "$local_dir/" "${location[@]}"; then
    keep_vm=true
    echo "Retry the copy from $vm:$remote_dir/result into $local_dir/" >&2
    exit 1
fi
if (( status != 0 )); then
    echo "Experiment failed. Retrieved outputs and console log under $local_dir" >&2
    exit "$status"
fi
for file in pytorch_lora_weights.safetensors before.png after.png trackio-run.json; do
    test -s "$local_dir/result/$file"
done
test -d "$local_dir/result/trackio"
echo "Adapter, images, Trackio run and console log: $local_dir"
echo "TRACKIO_DIR=\"\$PWD/$local_dir/result/trackio\" uvx --from trackio==0.45.1 trackio show --project dreambooth-lora-sd-xl"
