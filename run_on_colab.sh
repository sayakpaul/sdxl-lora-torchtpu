#!/usr/bin/env bash
# Run on the workstation authenticated to Colab. Each invocation owns one session.
set -euo pipefail
cd "$(dirname "$0")"
colab=(uvx --python 3.12 --from google-colab-cli==0.8.0 colab --auth adc)
chip=${COLAB_TPU:-v6e1}
case "$chip" in
    v6e1|v5e1) ;;
    *) echo "COLAB_TPU must be v6e1 or v5e1" >&2; exit 2 ;;
esac
session="sdxl-$(date -u +%Y%m%d-%H%M%S)-$RANDOM"
local_dir="runs/$session"
mkdir -p "$local_dir"

# TPU provisioning can take longer than the client's 120-second read timeout.
COLAB_CLI_REQUEST_TIMEOUT=${COLAB_CLI_REQUEST_TIMEOUT:-600} \
    "${colab[@]}" new -s "$session" --tpu "$chip"
stop_session() { "${colab[@]}" stop -s "$session"; }
trap stop_session EXIT
"${colab[@]}" upload -s "$session" run_sdxl_lora.py content/run_sdxl_lora.py
"${colab[@]}" upload -s "$session" sdxl-torchtpu.patch content/sdxl-torchtpu.patch

status=0
"${colab[@]}" exec -s "$session" --timeout 7200 <<PY 2>&1 | tee "$local_dir/console.log" || status=$?
import shutil
import subprocess
import sys
import os
from pathlib import Path

work = Path("/content")
result = work / "result"
result.mkdir()
def run_logged(command):
    # Forward child output through IPython so the Colab CLI can capture it.
    with (result / "experiment.log").open("a") as log, subprocess.Popen(
        command, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"}, text=True, errors="replace",
    ) as child:
        for line in child.stdout:
            log.write(line)
            print(line, end="", flush=True)
        if child.wait():
            raise subprocess.CalledProcessError(child.returncode, command)

try:
    if not shutil.which("uv"):
        subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
    cmd = ["uv", "run", "--python", "3.12", "run_sdxl_lora.py", "--output", "result"]
    train_flags = ["--gradient-checkpointing"] if "$chip" == "v5e1" else []
    print("Training on requested Colab TPU $chip", flush=True)
    run_logged([*cmd, *train_flags])
    run_logged([*cmd, "--preview-only"])
finally:
    shutil.make_archive(str(work / "sdxl-result"), "gztar", root_dir=work, base_dir="result")
PY

if ! "${colab[@]}" download -s "$session" content/sdxl-result.tar.gz "$local_dir/result.tar.gz"; then
    # Keep the only remote copy available for recovery.
    trap - EXIT
    echo "Download failed. The Colab session $session is still allocated." >&2
    printf 'Retry: %q ' "${colab[0]}" >&2
    printf '%q ' "${colab[@]:1}" download -s "$session" content/sdxl-result.tar.gz "$local_dir/result.tar.gz" >&2
    printf '\n' >&2
    echo "After recovery, release the session with: ${colab[*]} stop -s $session" >&2
    exit 1
fi
tar -xzf "$local_dir/result.tar.gz" -C "$local_dir"
if (( status != 0 )); then
    echo "Remote execution failed. See $local_dir/console.log and the retrieved partial results." >&2
    exit "$status"
fi
for file in pytorch_lora_weights.safetensors before.png after.png trackio-run.json; do
    test -s "$local_dir/result/$file"
done
test -d "$local_dir/result/trackio"
echo "Results: $local_dir/result"
echo "TRACKIO_DIR=\"\$PWD/$local_dir/result/trackio\" uvx --from trackio==0.45.1 trackio show --project dreambooth-lora-sd-xl"
