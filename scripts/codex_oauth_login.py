#!/usr/bin/env python3
"""Run the OpenAI Codex device-code login and persist the DSA credential.

Authorizes a ChatGPT/Codex subscription for ``GENERATION_BACKEND=codex_oauth``.
Prints a verification URL plus a device code, waits for the browser
confirmation, then writes the OAuth bundle to the configured path. Run it once
per deployment; the backend refreshes the token on its own afterwards.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import codex_oauth  # noqa: E402


def _default_auth_file() -> str:
    return (
        os.getenv("CODEX_OAUTH_AUTH_FILE", "").strip()
        or codex_oauth.DEFAULT_AUTH_FILE
    )


def _print_prompt(device: Dict[str, Any]) -> None:
    print()
    print(f"  在浏览器中打开：{device['verification_url']}")
    print(f"  输入设备码：    {device['user_code']}")
    print()
    print("  等待授权中，完成后本终端会自动继续（Ctrl-C 取消）…", flush=True)


def _print_status(auth_file: str) -> int:
    try:
        credential = codex_oauth.load_credential(auth_file)
    except codex_oauth.CodexOAuthError as exc:
        print(f"  {exc.detail or exc.reason}")
        return 1

    remaining = float(credential.get("expires_at") or 0) - time.time()
    print(f"  账号      {credential.get('email', '-')}")
    print(f"  套餐      {credential.get('plan_type', '-')}")
    print(f"  凭证文件  {os.path.expanduser(auth_file)}")
    if remaining > 0:
        print(f"  token     有效，{int(remaining / 60)} 分钟后过期（调用时自动刷新）")
    else:
        print("  token     已过期，下次调用时自动刷新")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Authorize a ChatGPT/Codex subscription for GENERATION_BACKEND=codex_oauth "
            "using the device-code flow. Run once on any machine that can reach a browser."
        )
    )
    parser.add_argument(
        "--auth-file",
        default=_default_auth_file(),
        help="凭证写入路径（默认读取 CODEX_OAUTH_AUTH_FILE，回退 %(default)s）",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="只查看当前凭证状态，不重新登录",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="已有有效凭证时也重新登录",
    )
    args = parser.parse_args()

    auth_file = args.auth_file
    if args.status:
        return _print_status(auth_file)

    if not args.force:
        try:
            existing = codex_oauth.load_credential(auth_file)
        except codex_oauth.CodexOAuthError:
            existing = None
        if existing and float(existing.get("expires_at") or 0) > time.time():
            expires = datetime.fromtimestamp(
                float(existing["expires_at"]), tz=timezone.utc
            ).isoformat()
            print(f"  已存在有效凭证（{existing.get('email', '-')}，{expires} 到期）")
            print("  如需重新登录请加 --force")
            return 0

    try:
        credential = codex_oauth.device_login(auth_file, on_prompt=_print_prompt)
    except codex_oauth.CodexOAuthError as exc:
        print(f"\n  授权失败：{exc.message}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n  已取消", file=sys.stderr)
        return 130

    print()
    print(f"  授权成功  {credential.get('email', '-')}（{credential.get('plan_type', '-')}）")
    print(f"  凭证已写入 {os.path.expanduser(auth_file)}")
    print()
    print("  接下来在 .env 中设置：GENERATION_BACKEND=codex_oauth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
