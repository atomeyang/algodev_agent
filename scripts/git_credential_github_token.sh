#!/usr/bin/env bash
# Git credential helper that answers HTTPS prompts for github.com with a token
# stored in a local file. The file is created by scripts/github_token_setup.sh
# and lives outside the repository, so the token is never committed.
#
# Invoked by git as: git_credential_github_token.sh <get|store|erase>
set -euo pipefail

[ "${1:-get}" = "get" ] || exit 0

token_file="${GITHUB_TOKEN_FILE:-$HOME/.config/algo_agent/github_token}"
[ -r "$token_file" ] || exit 0

host=""
while IFS= read -r line; do
  case "$line" in
    "") break ;;
    host=*) host="${line#host=}" ;;
  esac
done || true

case "$host" in
  ""|github.com) ;;
  *) exit 0 ;;
esac

token="$(tr -d '[:space:]' < "$token_file")"
[ -n "$token" ] || exit 0

username="${GITHUB_USERNAME:-$(git config --get github.user 2>/dev/null || true)}"
[ -n "$username" ] || username="x-access-token"

printf 'username=%s\npassword=%s\n' "$username" "$token"
