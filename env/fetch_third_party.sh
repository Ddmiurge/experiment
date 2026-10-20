#!/usr/bin/env bash
# 拉取并锁定第三方仓库。
#
# 设计：不在脚本里写死 commit SHA（写死了就没人去更新，反而容易过期）。
# 改成首次拉取时解析 ref → SHA，写进 third_party.lock；之后所有机器
# 一律按 lock 里的 SHA 检出。lock 文件进 git，是复现依据。
#
# 用法：
#     bash env/fetch_third_party.sh          # 按 lock 检出（没有则首次解析）
#     bash env/fetch_third_party.sh --relock # 忽略 lock 重新解析（慎用）
#
# 注意：third_party/ 里的仓库一律不改。所有适配写在 src/pruning/ 外层。

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="$ROOT/third_party"
LOCK="$ROOT/third_party.lock"
RELOCK=0
[ "${1:-}" = "--relock" ] && RELOCK=1

mkdir -p "$TP"
[ -f "$LOCK" ] || printf '# name\turl\tsha\tfetched_at\n' > "$LOCK"

ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }

# name|url|ref
#
# ⚠ 下面的 URL 我没有逐个核验过。clone 失败会立刻报错并继续下一个，
#   第一次跑请盯输出，把失败的 URL 改对再跑一遍。
#
# ⚠ TRIM 那一行是 TODO：Token-Pruning 用的 TRIM 是哪个仓库需要先确认
#   （见 代码任务清单 避坑清单）。T1.2 之前必须填上。
REPOS=(
  "llava|https://github.com/haotian-liu/LLaVA|main"
  "trim|TODO|TODO"
  "sparsevlm|https://github.com/Gumpest/SparseVLMs|main"
  "mm-safetybench|https://github.com/ThuCCSLab/MM-SafetyBench|main"
)

locked_sha() {  # $1=name
  [ -f "$LOCK" ] || return 1
  awk -v n="$1" '$1==n {print $3; exit}' "$LOCK"
}

record_lock() {  # $1=name $2=url $3=sha
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$(date -Iseconds)" >> "$LOCK"
}

for entry in "${REPOS[@]}"; do
  IFS='|' read -r name url ref <<< "$entry"

  if [ "$url" = "TODO" ] || [ "$ref" = "TODO" ]; then
    warn "$name —— URL/ref 未填，跳过（待核验）"
    continue
  fi

  sha=""
  [ "$RELOCK" -eq 0 ] && sha="$(locked_sha "$name" || true)"

  if [ -z "$sha" ]; then
    sha="$(git ls-remote "$url" "$ref" 2>/dev/null | awk '{print $1}' | head -1)"
    if [ -z "$sha" ]; then
      bad "$name —— 无法从 $url 解析 $ref（URL 错？出网不通？）"
      continue
    fi
    record_lock "$name" "$url" "$sha"
    warn "$name —— 首次解析，已写入 third_party.lock：${sha:0:12}"
  fi

  dest="$TP/$name"
  if [ -d "$dest/.git" ]; then
    if git -C "$dest" fetch --quiet origin "$sha" 2>/dev/null; then
      git -C "$dest" checkout --quiet "$sha" 2>/dev/null \
        && ok "$name @ ${sha:0:12}" \
        || bad "$name —— checkout $sha 失败"
    else
      bad "$name —— fetch $sha 失败（lock 里的 SHA 可能已不存在）"
    fi
  else
    if git clone --quiet "$url" "$dest" 2>/dev/null; then
      git -C "$dest" checkout --quiet "$sha" 2>/dev/null \
        && ok "$name @ ${sha:0:12}（新克隆）" \
        || bad "$name —— clone 成功但 checkout $sha 失败"
    else
      bad "$name —— clone $url 失败"
    fi
  fi
done

printf '\n锁定状态：\n'
[ -f "$LOCK" ] && sed 's/^/  /' "$LOCK"
printf '\nthird_party.lock 要进 git —— 它才是复现依据，仓库内容不是。\n'
