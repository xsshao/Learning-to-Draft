#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: bash scripts/submit_qwen3_32b_chain.sh <size|depth> [options] [-- extra train args...]

Submit one or more chained Qwen3-32B jobs that auto-resume from the newest
checkpoint after each timeout.

Options:
  --chunks N            Number of jobs to chain (default: 1)
  --time HH:MM:SS       Walltime per job (default: 01:00:00)
  --partition NAME      Override the partition
  --account NAME        Override the account
  --gres SPEC           Override the GRES request (example: gpu:RTXA6000:1)
  --constraint NAME     Override the constraint
  --exclude NODELIST    Override the exclude list
  --checkpoint-freq N   Save frequency to export to the job (default: 5000)
  --gpu-index IDX       GPU index to export to the job (default: 0)
  --dry-run             Print the sbatch commands without submitting
  -h, --help            Show this help message

Examples:
  bash scripts/submit_qwen3_32b_chain.sh size --chunks 4
  bash scripts/submit_qwen3_32b_chain.sh depth --chunks 8 --partition secondary --gres gpu:RTXA6000:1
  bash scripts/submit_qwen3_32b_chain.sh size --chunks 4 --partition h100 --gres gpu:H100:1
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

policy="$1"
shift

case "${policy}" in
    size|depth)
        ;;
    *)
        echo "Unknown policy: ${policy}" >&2
        usage
        exit 1
        ;;
esac

chunks=1
time_limit="01:00:00"
partition=""
account=""
gres=""
constraint=""
exclude=""
checkpoint_freq=5000
gpu_index=0
dry_run=0
train_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --chunks)          chunks="$2";           shift 2 ;;
        --time)            time_limit="$2";       shift 2 ;;
        --partition)       partition="$2";        shift 2 ;;
        --account)         account="$2";          shift 2 ;;
        --gres)            gres="$2";             shift 2 ;;
        --constraint)      constraint="$2";       shift 2 ;;
        --exclude)         exclude="$2";          shift 2 ;;
        --checkpoint-freq) checkpoint_freq="$2";  shift 2 ;;
        --gpu-index)       gpu_index="$2";        shift 2 ;;
        --dry-run)         dry_run=1;             shift 1 ;;
        --)                shift; train_args=("$@"); break ;;
        -h|--help)         usage; exit 0 ;;
        *)                 train_args+=("$1");    shift 1 ;;
    esac
done

if ! [[ "${chunks}" =~ ^[0-9]+$ ]] || (( chunks < 1 )); then
    echo "--chunks must be a positive integer." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sbatch_script="${REPO_ROOT}/slurm/train_qwen3_32b_${policy}.sbatch"

if [[ ! -f "${sbatch_script}" ]]; then
    echo "Missing sbatch script: ${sbatch_script}" >&2
    exit 1
fi

export_vars=(
    "ALL"
    "AUTO_RESUME=1"
    "CHECKPOINT_FREQ=${checkpoint_freq}"
    "GPU_INDEX=${gpu_index}"
)

join_by_comma() {
    local joined="$1"
    shift
    local item
    for item in "$@"; do
        joined+=",${item}"
    done
    printf '%s\n' "${joined}"
}

export_arg="$(join_by_comma "${export_vars[0]}" "${export_vars[@]:1}")"

job_ids=()
previous_job_id=""

for ((i = 1; i <= chunks; i++)); do
    sbatch_cmd=(sbatch --parsable --time="${time_limit}" --export="${export_arg}")

    if [[ -n "${partition}" ]]; then
        sbatch_cmd+=(--partition="${partition}")
    fi
    if [[ -n "${account}" ]]; then
        sbatch_cmd+=(--account="${account}")
    fi
    if [[ -n "${gres}" ]]; then
        sbatch_cmd+=(--gres="${gres}")
    fi
    if [[ -n "${constraint}" ]]; then
        sbatch_cmd+=(--constraint="${constraint}")
    fi
    if [[ -n "${exclude}" ]]; then
        sbatch_cmd+=(--exclude="${exclude}")
    fi
    if [[ -n "${previous_job_id}" ]]; then
        sbatch_cmd+=(--dependency="afterany:${previous_job_id}")
    fi

    sbatch_cmd+=("${sbatch_script}")
    if (( ${#train_args[@]} > 0 )); then
        sbatch_cmd+=("${train_args[@]}")
    fi

    if (( dry_run )); then
        printf 'DRY RUN [%d/%d]: ' "${i}" "${chunks}"
        printf '%q ' "${sbatch_cmd[@]}"
        printf '\n'
        previous_job_id="DRYRUN${i}"
    else
        job_id="$("${sbatch_cmd[@]}")"
        job_id="${job_id%%;*}"
        job_ids+=("${job_id}")
        previous_job_id="${job_id}"
        echo "Submitted ${policy} chain job ${i}/${chunks}: ${job_id}"
    fi
done

if (( dry_run )); then
    echo "Dry run complete."
else
    echo "Queued ${#job_ids[@]} ${policy} job(s): ${job_ids[*]}"
fi
