#!/usr/bin/env bash
set -euo pipefail

CONFIG="batch_datasets.json"
DATASETS=""
DRY_RUN=0
RUN_NAME=""
PROFILE=""
AC_EPOCHS="100"
AC_SIZE_SWEEP="200,400,600,800,1000,1200,1400"
TRAIN_ARGS=()
DATASET_ARGS=()

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
  --profile NAME              Named profile from config.profiles
  --eval-only                 Load checkpoint and skip training
  --search-only               Alias of --eval-only for search/reuse runs
  --model-name NAME           Model/checkpoint name for save and eval-only reuse
  --ckpt-path PATH            Checkpoint path passed to train_ig.py
  --ilssc-auto                Use dataset-specific ILSSC profile from batch_datasets.json
  --dataset-arg DATASET KEY VALUE
                              Override one train_ig.py arg for one dataset
  --acm-warmup N              ACM ILSSC warmup epochs
  --acm-ramp N                ACM ILSSC ramp epochs
  --dblp-warmup N             DBLP ILSSC warmup epochs
  --dblp-ramp N               DBLP ILSSC ramp epochs
  --imdb-warmup N             IMDB_NEW ILSSC warmup epochs
  --imdb-ramp N               IMDB_NEW ILSSC ramp epochs
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
  --ilssc-high-order-beta X   IDBR-inspired sparse high-order prior weight
  --ilssc-warmup-epochs N     Disable ILSSC for first N epochs
  --ilssc-ramp-epochs N       Linearly ramp ILSSC weight after warmup
  --id-ilssc                  Enable intent-distribution-aware ILSSC
  --intent-dist-k N           Number of ID-ILSSC intent prototypes
  --intent-dist-tau X         ID-ILSSC intent distribution temperature
  --intent-dist-beta X        ID-ILSSC distribution similarity weight
  --intent-dist-proto-mode M  ID-ILSSC prototype mode: random or score
  --intent-dist-stable        Enable stable confidence-gated intent distribution memory
  --intent-dist-update-interval N
                              Stable intent distribution memory update interval
  --intent-dist-memory-warmup N
                              Stable intent distribution memory warmup epochs
  --intent-dist-ema X         EMA coefficient for stable intent prototypes
  --intent-dist-anchor-pool N Anchor pool size for stable prototype sampling; 0=all nodes
  --intent-dist-conf-tau X    Confidence gate temperature
  --intent-dist-min-conf X    Minimum confidence product for gate
  --greedy-init-seed-size N   Greedy initial seed size
  --greedy-init-seed-hops N   Greedy initial seed hops
  --greedy-high-order-beta X  HSE-Greedy query-candidate high-order reachability weight
  --greedy-comm-cohesion-beta X
                              HSE-Greedy candidate-to-community halo cohesion weight
  --greedy-comm-direct-beta X HSE-Greedy candidate-to-current-community direct cohesion weight
  --greedy-boundary-gamma X   HSE-Greedy boundary expansion penalty weight
  --greedy-patience N         Stop after N non-improving greedy steps; 0=first drop
  --greedy-min-gain-tol X     Minimum greedy F1 gain tolerance
  --greedy-balance-alpha X    Mean-similarity support bonus for greedy trace stop and prefix selection
  --greedy-adaptive-cap-alpha X
                              Scale the per-query adaptive greedy cap; 0 disables adaptation
  --greedy-adaptive-cap-floor N
                              Minimum per-query greedy cap before scaling
  --greedy-trace-cap-ratio X  Trace budget as a multiple of the final cap when multiple w values are evaluated
  --greedy-hse-normalize      Normalize HSE structural terms inside candidate pool
  --greedy-hse-density        Use HSE-adjusted utility for greedy density selection
  --greedy-recall-expand-size N
                              Add up to N high-order frontier nodes after core community selection
  --greedy-recall-min-sim-delta X
                              Fallback candidate min similarity = avg_sim + X
  -h, --help                  Show this help

Examples:
  bash run_batch.sh
  bash run_batch.sh -d ACM,DBLP
  bash run_batch.sh --dry-run -d ACM --rl
  bash run_batch.sh -d IMDB_NEW --rl --ac-size-sweep 200,400,800,1200,1600
  bash run_batch.sh -d ACM --frontier-ic-spnm 0.003
  bash run_batch.sh -d ACM,DBLP,IMDB_NEW --relation-fusion transformer
  bash run_batch.sh --ilssc-auto -d ACM,DBLP,IMDB_NEW
  bash run_batch.sh --ilssc-auto -d ACM,DBLP,IMDB_NEW --acm-warmup 20 --acm-ramp 40 --dblp-warmup 10 --dblp-ramp 30 --imdb-warmup 0 --imdb-ramp 0
  bash run_batch.sh --ilssc-auto -d ACM,DBLP,IMDB_NEW --dataset-arg ACM ilssc_seed_size 6 --dataset-arg DBLP ilssc_seed_size 8
  bash run_batch.sh --ilssc-auto --id-ilssc -d ACM,DBLP,IMDB_NEW --intent-dist-k 16 --intent-dist-beta 0.5
  bash run_batch.sh --run-name scid_ilssc --profile scid_ilssc_auto -d ACM,DBLP,IMDB_NEW --intent-dist-beta 0.2 --intent-dist-stable
  bash run_batch.sh --run-name hidbr_ilssc --profile hidbr_ilssc_auto -d ACM,DBLP,IMDB_NEW --ilssc-high-order-beta 0.2
  bash run_batch.sh --run-name hse_greedy --profile hidbr_ilssc_auto -d ACM,DBLP,IMDB_NEW --ilssc-high-order-beta 0.2 --greedy-high-order-beta 0.2 --greedy-comm-direct-beta 0.1 --greedy-comm-cohesion-beta 0.05 --greedy-boundary-gamma 0.03 --greedy-hse-pool-size 512 --greedy-recall-expand-size 128
  bash run_batch.sh --run-name hse_cap --profile hse_greedy_auto --eval-only -d ACM,DBLP,IMDB_NEW --greedy-hse-pool-size 64 --greedy-max-size 512
  bash run_batch.sh --run-name hse_eval_only --profile hse_greedy_auto --eval-only -d ACM,DBLP,IMDB_NEW --greedy-hse-pool-size 64
  bash run_batch.sh --run-name hse_save --profile hse_greedy_auto --model-name hse64 -d ACM,DBLP,IMDB_NEW
  bash run_batch.sh --run-name ilssc_auto -d ACM,DBLP,IMDB_NEW --profile ilssc_auto
  bash run_batch.sh --run-name ilssc_seed -d ACM,DBLP,IMDB_NEW --lambda-ilssc 0.1 --ilssc-seed-size 8 --greedy-init-seed-size 4 --greedy-init-seed-hops 2
  bash run_batch.sh --run-name ilssc_seed -d ACM,DBLP,IMDB_NEW -- --lambda_ilssc 0.1 --greedy_init_seed_size 4
  bash run_batch.sh -d DBLP -- --cs_relations apa --cs_w_list 0.10,0.12,0.14,0.16
EOF
}

add_train_arg() {
  TRAIN_ARGS+=("--train_arg=$1")
}

add_dataset_arg() {
  DATASET_ARGS+=("--dataset_arg" "$1" "$2" "$3")
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
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --eval-only|--eval_only|--search-only|--search_only)
      add_train_arg "--eval_only"
      shift
      ;;
    --model-name|--model_name)
      add_train_arg "--model_name"
      add_train_arg "$2"
      shift 2
      ;;
    --ckpt-path|--ckpt_path)
      add_train_arg "--ckpt_path"
      add_train_arg "$2"
      shift 2
      ;;
    --ilssc-auto|--ilssc_auto)
      PROFILE="ilssc_auto"
      if [[ -z "${RUN_NAME}" ]]; then
        RUN_NAME="ilssc_auto"
      fi
      shift
      ;;
    --dataset-arg|--dataset_arg)
      add_dataset_arg "$2" "$3" "$4"
      shift 4
      ;;
    --acm-warmup|--acm_warmup)
      add_dataset_arg "ACM" "ilssc_warmup_epochs" "$2"
      shift 2
      ;;
    --acm-ramp|--acm_ramp)
      add_dataset_arg "ACM" "ilssc_ramp_epochs" "$2"
      shift 2
      ;;
    --dblp-warmup|--dblp_warmup)
      add_dataset_arg "DBLP" "ilssc_warmup_epochs" "$2"
      shift 2
      ;;
    --dblp-ramp|--dblp_ramp)
      add_dataset_arg "DBLP" "ilssc_ramp_epochs" "$2"
      shift 2
      ;;
    --imdb-warmup|--imdb_warmup|--imdb-new-warmup|--imdb_new_warmup)
      add_dataset_arg "IMDB_NEW" "ilssc_warmup_epochs" "$2"
      shift 2
      ;;
    --imdb-ramp|--imdb_ramp|--imdb-new-ramp|--imdb_new_ramp)
      add_dataset_arg "IMDB_NEW" "ilssc_ramp_epochs" "$2"
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
    --ilssc-high-order-beta|--ilssc_high_order_beta)
      add_train_arg "--ilssc_high_order_beta"
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
    --id-ilssc|--id_ilssc|--ilssc-use-intent-dist|--ilssc_use_intent_dist)
      add_train_arg "--ilssc_use_intent_dist"
      shift
      ;;
    --intent-dist-k|--intent_dist_k)
      add_train_arg "--intent_dist_k"
      add_train_arg "$2"
      shift 2
      ;;
    --intent-dist-tau|--intent_dist_tau)
      add_train_arg "--intent_dist_tau"
      add_train_arg "$2"
      shift 2
      ;;
    --intent-dist-beta|--intent_dist_beta)
      add_train_arg "--intent_dist_beta"
      add_train_arg "$2"
      shift 2
      ;;
    --intent-dist-proto-mode|--intent_dist_proto_mode)
      add_train_arg "--intent_dist_proto_mode"
      add_train_arg "$2"
      shift 2
      ;;
    --intent-dist-stable|--intent_dist_stable|--scid-ilssc|--scid_ilssc)
      add_train_arg "--intent_dist_stable"
      shift
      ;;
    --intent-dist-update-interval|--intent_dist_update_interval)
      add_train_arg "--intent_dist_update_interval"
      add_train_arg "$2"
      shift 2
      ;;
    --intent-dist-memory-warmup|--intent_dist_memory_warmup)
      add_train_arg "--intent_dist_memory_warmup"
      add_train_arg "$2"
      shift 2
      ;;
    --intent-dist-ema|--intent_dist_ema)
      add_train_arg "--intent_dist_ema"
      add_train_arg "$2"
      shift 2
      ;;
    --intent-dist-anchor-pool|--intent_dist_anchor_pool)
      add_train_arg "--intent_dist_anchor_pool"
      add_train_arg "$2"
      shift 2
      ;;
    --intent-dist-conf-tau|--intent_dist_conf_tau)
      add_train_arg "--intent_dist_conf_tau"
      add_train_arg "$2"
      shift 2
      ;;
    --intent-dist-min-conf|--intent_dist_min_conf)
      add_train_arg "--intent_dist_min_conf"
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
    --greedy-high-order-beta|--greedy_high_order_beta|--hse-high-order-beta|--hse_high_order_beta)
      add_train_arg "--greedy_high_order_beta"
      add_train_arg "$2"
      shift 2
      ;;
    --greedy-comm-cohesion-beta|--greedy_comm_cohesion_beta|--hse-comm-cohesion-beta|--hse_comm_cohesion_beta)
      add_train_arg "--greedy_comm_cohesion_beta"
      add_train_arg "$2"
      shift 2
      ;;
    --greedy-comm-direct-beta|--greedy_comm_direct_beta|--hse-comm-direct-beta|--hse_comm_direct_beta)
      add_train_arg "--greedy_comm_direct_beta"
      add_train_arg "$2"
      shift 2
      ;;
    --greedy-boundary-gamma|--greedy_boundary_gamma|--hse-boundary-gamma|--hse_boundary_gamma)
      add_train_arg "--greedy_boundary_gamma"
      add_train_arg "$2"
      shift 2
      ;;
    --greedy-patience|--greedy_patience)
      add_train_arg "--greedy_patience"
      add_train_arg "$2"
      shift 2
      ;;
    --greedy-min-gain-tol|--greedy_min_gain_tol)
      add_train_arg "--greedy_min_gain_tol"
      add_train_arg "$2"
      shift 2
      ;;
    --greedy-balance-alpha|--greedy_balance_alpha)
      add_train_arg "--greedy_balance_alpha"
      add_train_arg "$2"
      shift 2
      ;;
    --greedy-adaptive-cap-alpha|--greedy_adaptive_cap_alpha)
      add_train_arg "--greedy_adaptive_cap_alpha"
      add_train_arg "$2"
      shift 2
      ;;
    --greedy-adaptive-cap-floor|--greedy_adaptive_cap_floor)
      add_train_arg "--greedy_adaptive_cap_floor"
      add_train_arg "$2"
      shift 2
      ;;
    --greedy-trace-cap-ratio|--greedy_trace_cap_ratio)
      add_train_arg "--greedy_trace_cap_ratio"
      add_train_arg "$2"
      shift 2
      ;;
    --greedy-max-size|--greedy_max_size)
      add_train_arg "--greedy_max_size"
      add_train_arg "$2"
      shift 2
      ;;
    --greedy-hse-pool-size|--greedy_hse_pool_size|--hse-pool-size|--hse_pool_size)
      add_train_arg "--greedy_hse_pool_size"
      add_train_arg "$2"
      shift 2
      ;;
    --greedy-hse-normalize|--greedy_hse_normalize|--hse-normalize|--hse_normalize)
      add_train_arg "--greedy_hse_normalize"
      shift
      ;;
    --greedy-hse-density|--greedy_hse_density|--hse-density|--hse_density)
      add_train_arg "--greedy_hse_density"
      shift
      ;;
    --greedy-recall-expand-size|--greedy_recall_expand_size|--hse-recall-expand-size|--hse_recall_expand_size)
      add_train_arg "--greedy_recall_expand_size"
      add_train_arg "$2"
      shift 2
      ;;
    --greedy-recall-min-sim-delta|--greedy_recall_min_sim_delta|--hse-recall-min-sim-delta|--hse_recall_min_sim_delta)
      add_train_arg "--greedy_recall_min_sim_delta"
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
if [[ -n "${PROFILE}" ]]; then
  CMD+=(--profile "${PROFILE}")
fi
CMD+=("${DATASET_ARGS[@]}")
CMD+=("${TRAIN_ARGS[@]}")

"${CMD[@]}"
