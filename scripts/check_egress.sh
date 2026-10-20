#!/usr/bin/env bash
# 探测服务器的出网边界。
#
# 这是整个计划里最先要跑的东西 —— 在它出结果之前，
# "代码怎么上去、结果怎么回来"两个问题都没有确定答案。
#
# 用法：
#     bash scripts/check_egress.sh [你的仓库URL]
#
# 例：
#     bash scripts/check_egress.sh https://github.com/<账号>/<repo>.git

set -uo pipefail

REPO_URL="${1:-}"

ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
hdr()  { printf '\n\033[1m%s\033[0m\n' "$1"; }

code_of() {  # $1=url -> http code, 失败给 000
  curl -sS -m 10 -o /dev/null -w '%{http_code}' "$1" 2>/dev/null || echo 000
}

hdr "A · 有没有出网"
c="$(code_of https://www.baidu.com)"
if [ "$c" = "200" ] || [ "$c" = "301" ] || [ "$c" = "302" ]; then
  ok "百度 $c —— 有出网"
else
  bad "百度 $c —— 基本出网都不通，后面全部免谈，先找管理员"
fi

hdr "B · 能不能 pull GitHub  ← 单点依赖"
c="$(code_of https://github.com)"
if [ "$c" = "200" ] || [ "$c" = "301" ]; then
  ok "github.com $c"
else
  bad "github.com $c"
fi
if [ -n "$REPO_URL" ]; then
  if git ls-remote --heads "$REPO_URL" >/dev/null 2>&1; then
    ok "git ls-remote $REPO_URL 通过 —— 代码能上去"
  else
    bad "git ls-remote $REPO_URL 失败 —— 代码上不去，见下方退路"
  fi
else
  warn "没给仓库 URL，跳过实测。补上参数重跑。"
fi

hdr "C · 能不能下模型"
for u in https://hf-mirror.com https://huggingface.co https://pypi.org/simple; do
  c="$(code_of "$u")"
  case "$c" in
    200|301|302) ok "$u $c" ;;
    *)           warn "$u $c" ;;
  esac
done

hdr "D · 出站是否只放行 GET"
# 这一条只用来判明"不能 push"是网络限制还是凭据限制。
# 不是用来绕规定的 —— 若测出能 POST 但规定不许 push，那是规定问题，按主方案走。
c="$(curl -sS -m 10 -X POST -d 't=1' -o /dev/null -w '%{http_code}' \
      https://httpbin.org/post 2>/dev/null || echo 000)"
case "$c" in
  200) ok "POST 出站通畅 $c —— 限制在凭据，不在网络" ;;
  000) bad "POST 被挡 $c —— 网络层只放行 GET" ;;
  *)   warn "POST 返回 $c —— 说不准，人工通道是唯一可靠方案" ;;
esac

hdr "结论怎么读"
cat <<'EOF'
  B 通过  → 代码走 git：Mac push → GitHub → 服务器 pull。主方案可用。
  B 失败  → 代码上不去。退路按顺序试：
              1. Gitee（国内可达性通常更好）
              2. 单文件打包粘贴：
                   Mac 上  tar czf - -C code src scripts env | base64
                   把输出贴给网页 Agent，它在服务器上 base64 -d | tar xzf
                 整个 src/ 一次粘贴完，比逐文件粘贴可靠得多。
  D 被挡  → 死心用人工通道（结果块 + 复制粘贴）。别浪费时间找自动上行。
  D 通畅  → 技术上可以自动上行，但先确认规定是否允许。
EOF
printf '\n'
