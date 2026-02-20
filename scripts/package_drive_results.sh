#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/package_drive_results.sh /content/drive/MyDrive/MALTbot-results/chgnet_run/results.json [RUN_NAME]

SRC_JSON="${1:-}"
RUN_NAME="${2:-chgnet_mp_e_form_route_b}"

if [[ -z "${SRC_JSON}" ]]; then
  echo "Usage: $0 <path-to-results.json> [run_name]"
  exit 1
fi

DATE_KST="$(TZ=Asia/Seoul date +%F)"
DEST_DIR="results/daily/${DATE_KST}/${RUN_NAME}"
mkdir -p "${DEST_DIR}"
cp "${SRC_JSON}" "${DEST_DIR}/results.json"

echo "Copied -> ${DEST_DIR}/results.json"
