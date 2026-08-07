#!/usr/bin/env bash
set -euo pipefail

CONFIG="batch_datasets.json"
DATASETS=""
DRY_RUN=0
RUN_NAME=""
AC_EPOCHS="100"
AC_SIZE_SWEEP="200,400,600,800,1000,1200,1400"
TRAIN_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash run_batch.sh [options] [-- extra train_ig.py args]

Options:
  -c, --config PATH           Batch config JSON. Default: batch_datasets.json
  -d, --datasets LIST         Comma-separated datasets, e.g. ACM,DBLP,IMDB_NEW
  -e, --epochs N              Override --num_epochs
  --dry-run                   Print commands without running training
  --run-name NAME             Batch log directory name prefix under batch_runs/
  --rl                        Enable actor-critic search
  --ac-epochs N               Actor-critic epochs. Default: 100
  --ac-size-sweep LIST        Actor-critic size sweep list
  --ic-spnm WEIGHT            Enable IC-SPNM with this lambda
  --frontier-ic-spnm WEIGHT   Enable frontier-aware IC-SPNM mix mode
  --relation-fusion TYPE      Relation fusion type: icra or transformer
  --lambda-ilssc WEIGHT       Enable ILSSC with this lambda
  --ilssc-seed-size N         ILSSC seed size
  --ilssc-hard-pool N         ILSSC hard-negative candidate pool size
  --ilssc-neg-mode MODE       ILSSC negative mining: conservative or hard
  --ilssc-gate-mode MODE      ILSSC seed weighting: prior, intent, or mix
  --ilssc-warmup-epochs N     Disable ILSSC for first N epochs
  --ilssc-ramp-epochs N       Linearly ramp ILSSC weight after warmup
  --greedy-init-seed-size N   Greedy initial seed size
  --greedy-init-seed-hops N   Greedy initial seed hops
  -h, --help                  Show this help

Examples:
  bash run_batch.sh
  bash run_batch.sh -d ACM,DBLP
  bash run_batch.sh --dry-run -d ACM --rl
  bash run_batch.sh -d IMDB_NEW --rl --ac-size-sweep 200,400,800,1200,1600
  bash run_batch.sh -d ACM --frontier-ic-spnm 0.003
  bash run_batch.sh -d ACM,DBLP,IMDB_NEW --relation-fusion transformer
  bash run_batch.sh --run-name ilssc_seed -d ACM,DBLP,IMDB_NEW --lambda-ilssc 0.1 --ilssc-seed-size 8 --greedy-init-seed-size 4 --greedy-init-seed-hops 2
  bash run_batch.sh --run-name ilssc_seed -d ACM,DBLP,IMDB_NEW -- --lambda_ilssc 0.1 --greedy_init_seed_size 4
  bash run_batch.sh -d DBLP -- --cs_relations apa --cs_w_list 0.10,0.12,0.14,0.16
EOF
}

add_train_arg() {
  TRAIN_ARGS+=("--train_arg=$1")
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)
      CONFIG="$2"
      shift 2
      ;;
    -d|--datasets)
      DATASETS="$2"
      shift 2
      ;;
    -e|--epochs)
      add_train_arg "--num_epochs"
      add_train_arg "$2"
      shift 2
      ;;
    --dry-run|--dry_run)
      DRY_RUN=1
      shift
      ;;
    --run-name|--run_name|--name)
      RUN_NAME="$2"
      shift 2
      ;;
    --rl)
      add_train_arg "--use_actor_critic"
      add_train_arg "--ac_epochs"
      add_train_arg "${AC_EPOCHS}"
      add_train_arg "--ac_size_sweep"
      add_train_arg "${AC_SIZE_SWEEP}"
      shift
      ;;
    --ac-epochs)
      AC_EPOCHS="$2"
      add_train_arg "--ac_epochs"
      add_train_arg "$2"
      shift 2
      ;;
    --ac-size-sweep)
      AC_SIZE_SWEEP="$2"
      add_train_arg "--ac_size_sweep"
      add_train_arg "$2"
      shift 2
      ;;
    --ic-spnm)
      add_train_arg "--lambda_ic_spnm"
      add_train_arg "$2"
      shift 2
      ;;
    --frontier-ic-spnm)
      add_train_arg "--lambda_ic_spnm"
      add_train_arg "$2"
      add_train_arg "--ic_spnm_pos_mode"
      add_train_arg "mix"
      add_train_arg "--ic_spnm_frontier_ratio"
      add_train_arg "0.5"
      add_train_arg "--ic_spnm_frontier_hops"
      add_train_arg "2"
      add_train_arg "--ic_spnm_frontier_pool"
      add_train_arg "128"
      add_train_arg "--ic_spnm_frontier_conn_beta"
      add_train_arg "0.2"
      add_train_arg "--ic_spnm_frontier_min_align"
      add_train_arg "0.0"
      add_train_arg "--ic_spnm_num_queries"
      add_train_arg "4"
      add_train_arg "--ic_spnm_pos"
      add_train_arg "40"
      add_train_arg "--ic_spnm_neg"
      add_train_arg "128"
      add_train_arg "--ic_spnm_hard_pool"
      add_train_arg "1024"
      add_train_arg "--ic_spnm_intent_beta"
      add_train_arg "1.5"
      add_train_arg "--ic_spnm_struct_beta"
      add_train_arg "1.0"
      shift 2
      ;;
    --relation-fusion|--relation_fusion)
      add_train_arg "--relation_fusion"
      add_train_arg "$2"
      shift 2
      ;;
    --lambda-ilssc|--lambda_ilssc|--ilssc)
      add_train_arg "--lambda_ilssc"
      add_train_arg "$2"
      shift 2
      ;;
    --ilssc-seed-size|--ilssc_seed_size)
      add_train_arg "--ilssc_seed_size"
      add_train_arg "$2"
      shift 2
      ;;
    --ilssc-neg|--ilssc_neg)
      add_train_arg "--ilssc_neg"
      add_train_arg "$2"
      shift 2
      ;;
    --ilssc-hard-pool|--ilssc_hard_pool)
      add_train_arg "--ilssc_hard_pool"
      add_train_arg "$2"
      shift 2
      ;;
    --ilssc-num-queries|--ilssc_num_queries)
      add_train_arg "--ilssc_num_queries"
      add_train_arg "$2"
      shift 2
      ;;
    --ilssc-hops|--ilssc_hops)
      add_train_arg "--ilssc_hops"
      add_train_arg "$2"
      shift 2
      ;;
    --ilssc-neg-mode|--ilssc_neg_mode)
      add_train_arg "--ilssc_neg_mode"
      add_train_arg "$2"
      shift 2
      ;;
    --ilssc-gate-mode|--ilssc_gate_mode)
      add_train_arg "--ilssc_gate_mode"
      add_train_arg "$2"
      shift 2
      ;;
    --ilssc-warmup-epochs|--ilssc_warmup_epochs)
      add_train_arg "--ilssc_warmup_epochs"
      add_train_arg "$2"
      shift 2
      ;;
    --ilssc-ramp-epochs|--ilssc_ramp_epochs)
      add_train_arg "--ilssc_ramp_epochs"
      add_train_arg "$2"
      shift 2
      ;;
    --greedy-init-seed-size|--greedy_init_seed_size|--seed-greedy)
      add_train_arg "--greedy_init_seed_size"
      add_train_arg "$2"
      shift 2
      ;;
    --greedy-init-seed-hops|--greedy_init_seed_hops)
      add_train_arg "--greedy_init_seed_hops"
      add_train_arg "$2"
      shift 2
      ;;
    --greedy-init-seed-conn-beta|--greedy_init_seed_conn_beta)
      add_train_arg "--greedy_init_seed_conn_beta"
      add_train_arg "$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        add_train_arg "$1"
        shift
      done
      ;;
    *)
      if [[ "$1" == *.json && "${CONFIG}" == "batch_datasets.json" ]]; then
        CONFIG="$1"
      elif [[ -z "${DATASETS}" ]]; then
        DATASETS="$1"
      else
        echo "Unknown argument: $1" >&2
        usage >&2
        exit 2
      fi
      shift
      ;;
  esac
done

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

CMD=(python run_batch.py --config "${CONFIG}")
if [[ -n "${DATASETS}" ]]; then
  CMD+=(--datasets "${DATASETS}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=(--dry_run)
fi
if [[ -n "${RUN_NAME}" ]]; then
  CMD+=(--run_name "${RUN_NAME}")
fi
CMD+=("${TRAIN_ARGS[@]}")

"${CMD[@]}"
