#!/usr/bin/env bash
# Store a GitHub personal access token for this checkout.
#
# The token is read with hidden input and written to a file outside the
# repository (default: ~/.config/algo_agent/github_token, mode 600). A
# repository-local credential helper (scripts/git_credential_github_token.sh)
# then serves it to git, so `git fetch` / `git push` over HTTPS stop asking for
# a password. Run it again to rotate the token.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
token_file="${GITHUB_TOKEN_FILE:-$HOME/.config/algo_agent/github_token}"

mkdir -p "$(dirname "$token_file")"
chmod 700 "$(dirname "$token_file")"

printf 'GitHub token for %s\n' "$(git -C "$repo_root" remote get-url origin 2>/dev/null || echo 'this repository')"
printf 'Needs repo contents read/write (classic PAT with "repo" scope, or fine-grained).\n'
printf 'Paste it now, input is hidden: '
IFS= read -r -s token
printf '\n'

token="$(printf '%s' "$token" | tr -d '[:space:]')"
if [ -z "$token" ]; then
  printf 'No token entered, nothing was changed.\n' >&2
  exit 1
fi

umask 077
printf '%s' "$token" > "$token_file"
chmod 600 "$token_file"
printf 'Saved token to %s\n' "$token_file"

git -C "$repo_root" config --local credential.helper "$script_dir/git_credential_github_token.sh"
printf 'Configured credential helper for %s\n' "$repo_root"
