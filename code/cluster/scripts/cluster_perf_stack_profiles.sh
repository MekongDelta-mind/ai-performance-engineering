#!/usr/bin/env bash

# Shared helpers for stack-profile defaults used by FP4/grouped-GEMM scripts.
# shellcheck shell=bash

cluster_perf_profiles_json() {
  local root_dir="${1:-}"
  if [[ -n "${CLUSTER_PERF_STACK_PROFILES_JSON:-}" ]]; then
    printf '%s\n' "${CLUSTER_PERF_STACK_PROFILES_JSON}"
    return 0
  fi
  printf '%s/configs/cluster_perf_stack_profiles.json\n' "${root_dir}"
}

cluster_perf_profile_exists() {
  local root_dir="$1"
  local profile="$2"
  python3 - "$root_dir" "$profile" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
profile = sys.argv[2]
cfg = pathlib.Path(root / "configs" / "cluster_perf_stack_profiles.json")
data = json.loads(cfg.read_text(encoding="utf-8"))
sys.exit(0 if profile in (data.get("profiles") or {}) else 1)
PY
}

cluster_perf_profile_runtime_allowed() {
  local root_dir="$1"
  local profile="$2"
  local runtime="$3"
  python3 - "$root_dir" "$profile" "$runtime" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
profile = sys.argv[2]
runtime = sys.argv[3]
cfg = pathlib.Path(root / "configs" / "cluster_perf_stack_profiles.json")
data = json.loads(cfg.read_text(encoding="utf-8"))
profiles = data.get("profiles") or {}
entry = profiles.get(profile) or {}
allowed = set(entry.get("allowed_runtimes") or [])
sys.exit(0 if runtime in allowed else 1)
PY
}

cluster_perf_default_profile_for_runtime() {
  local root_dir="$1"
  local runtime="$2"
  python3 - "$root_dir" "$runtime" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
runtime = sys.argv[2]
cfg = pathlib.Path(root / "configs" / "cluster_perf_stack_profiles.json")
data = json.loads(cfg.read_text(encoding="utf-8"))
profiles = data.get("profiles") or {}
canonical = str(data.get("canonical_profile") or "")
if canonical and runtime in set((profiles.get(canonical) or {}).get("allowed_runtimes") or []):
    print(canonical)
    raise SystemExit(0)
for name, entry in profiles.items():
    if runtime in set(entry.get("allowed_runtimes") or []):
        print(name)
        raise SystemExit(0)
raise SystemExit(f"no stack profile supports runtime={runtime!r}")
PY
}

cluster_perf_profile_image_ref() {
  local root_dir="$1"
  local profile="$2"
  python3 - "$root_dir" "$profile" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
profile = sys.argv[2]
cfg = pathlib.Path(root / "configs" / "cluster_perf_stack_profiles.json")
data = json.loads(cfg.read_text(encoding="utf-8"))
profiles = data.get("profiles") or {}
entry = profiles.get(profile) or {}
print(str(entry.get("image_ref") or ""))
PY
}

cluster_perf_profile_math_allow_tf32() {
  local root_dir="$1"
  local profile="$2"
  python3 - "$root_dir" "$profile" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
profile = sys.argv[2]
cfg = pathlib.Path(root / "configs" / "cluster_perf_stack_profiles.json")
data = json.loads(cfg.read_text(encoding="utf-8"))
profiles = data.get("profiles") or {}
entry = profiles.get(profile) or {}
policy = entry.get("math_policy") or {}
allow = bool(policy.get("allow_tf32"))
print("1" if allow else "0")
PY
}

cluster_perf_profile_math_precision() {
  local root_dir="$1"
  local profile="$2"
  python3 - "$root_dir" "$profile" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
profile = sys.argv[2]
cfg = pathlib.Path(root / "configs" / "cluster_perf_stack_profiles.json")
data = json.loads(cfg.read_text(encoding="utf-8"))
profiles = data.get("profiles") or {}
entry = profiles.get(profile) or {}
policy = entry.get("math_policy") or {}
print(str(policy.get("float32_matmul_precision") or "high"))
PY
}
