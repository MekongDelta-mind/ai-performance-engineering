#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <node1_file> <node2_file>" >&2
  exit 1
fi

node1="$1"
node2="$2"

if [[ ! -f "$node1" || ! -f "$node2" ]]; then
  echo "Missing input files." >&2
  exit 1
fi

awk -F ' = ' '
  NR==FNR {a[$1]=$2; next}
  {b[$1]=$2}
  END {
    for (k in a) {
      if (!(k in b) || a[k] != b[k]) {
        printf("%s\n", k)
      }
    }
    for (k in b) {
      if (!(k in a)) printf("%s\n", k)
    }
  }' "$node1" "$node2" | sort -u > /tmp/tcp_diff_keys_12.txt

if [[ ! -s /tmp/tcp_diff_keys_12.txt ]]; then
  echo "No differing tcp sysctl keys across nodes." && exit 0
fi

echo "Differences in tcp sysctl keys:" 
while read -r key; do
  v1=$(grep -E "^${key} =" "$node1" | sed 's/^.*= //')
  v2=$(grep -E "^${key} =" "$node2" | sed 's/^.*= //')
  echo "${key}: node1=${v1} node2=${v2}"
done < /tmp/tcp_diff_keys_12.txt
