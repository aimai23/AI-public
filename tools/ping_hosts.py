#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
ツール名:
  ping_hosts.py

目的:
  ホスト名または IP アドレスの一覧ファイルを読み込み、下記の疎通テストを実行する。
  - ICMP (ping)
  - TCP (指定ポートへの connect)

  実行結果は「標準出力」と「ログファイル」の両方へ同時出力する。
  標準出力は簡潔表示、ログファイルは詳細表示とする。

保守設計:
  1. 設定値/終了コードを定数化し、仕様変更時の修正箇所を最小化する。
  2. ICMP/TCP を共通の並列実行基盤で扱い、実装重複を抑える。
  3. 結果表示順は入力順を維持し、運用確認を容易にする。
  4. エラーは可能な限り結果として集約し、途中停止を避ける。

終了コード:
  0: 全対象成功
  1: 1 件以上失敗
  2: 引数不正 / 入力ファイル不正 / 対象 0 件 / ログ初期化失敗
===============================================================================
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# 仕様定数
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COUNT = 1
DEFAULT_TIMEOUT = 2
DEFAULT_TCP_TIMEOUT = 2.0
DEFAULT_WORKERS = 20
DEFAULT_LOG_FILE = SCRIPT_DIR / "ping_hosts.log"
DEFAULT_LOG_FILE_DISPLAY = "tools/ping_hosts.log"

# 実行モード
MODE_ICMP = "icmp"
MODE_TCP = "tcp"
MODE_BOTH = "both"

# 終了コード
EXIT_ALL_SUCCESS = 0
EXIT_PARTIAL_FAILURE = 1
EXIT_INPUT_ERROR = 2

# 個別ターゲット実行時の疑似エラーコード
RETURNCODE_PING_NOT_FOUND = 127
RETURNCODE_UNEXPECTED_ERROR = 99

# Python 3.10 未満では dataclass(slots=True) が未対応のため互換分岐する。
DATACLASS_KWARGS = {"slots": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KWARGS)
class CheckTask:
    """
    実行ジョブ定義。

    protocol:
      "icmp" または "tcp"
    target:
      対象ホスト名/IP
    port:
      TCP の場合のみ使用
    """

    protocol: str
    target: str
    port: int | None = None


@dataclass(**DATACLASS_KWARGS)
class CheckResult:
    """
    1 ジョブ分の実行結果。

    protocol:
      "icmp" または "tcp"
    target:
      対象ホスト名/IP
    port:
      TCP の場合のみ使用
    returncode:
      実行結果コード（0 は成功）
    elapsed_seconds:
      実行時間（秒）
    output:
      実行時の詳細メッセージ
    command:
      実行コマンド文字列（ログ追跡用）
    """

    protocol: str
    target: str
    port: int | None
    returncode: int
    elapsed_seconds: float
    output: str
    command: str

    def endpoint(self) -> str:
        """
        表示用のエンドポイント文字列を返す。
        例:
          ICMP: host1
          TCP : host1:443
        """
        if self.port is None:
            return self.target
        return f"{self.target}:{self.port}"


def parse_args() -> argparse.Namespace:
    """
    CLI 引数定義。
    """
    parser = argparse.ArgumentParser(
        description="リストファイル内のホスト名/IP に対して ICMP/TCP 疎通テストを実行します。"
    )
    parser.add_argument(
        "list_file",
        type=Path,
        help="対象一覧ファイル（1 行 1 ホスト名/IP）。",
    )
    parser.add_argument(
        "--mode",
        choices=[MODE_ICMP, MODE_TCP, MODE_BOTH],
        default=MODE_ICMP,
        help="実行モード（既定: icmp）。",
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"ICMP: 1 対象あたりの送信回数（既定: {DEFAULT_COUNT}）。",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"ICMP: 1 回あたりの応答待ち秒数（既定: {DEFAULT_TIMEOUT}）。",
    )
    parser.add_argument(
        "--tcp-port",
        action="append",
        type=int,
        default=[],
        help="TCP: 疎通確認ポート（複数指定可）。例: --tcp-port 22 --tcp-port 443",
    )
    parser.add_argument(
        "--tcp-timeout",
        type=float,
        default=DEFAULT_TCP_TIMEOUT,
        help=f"TCP: connect タイムアウト秒（既定: {DEFAULT_TCP_TIMEOUT}）。",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"並列ワーカー数（既定: {DEFAULT_WORKERS}）。",
    )
    parser.add_argument(
        "--show-fail-output",
        action="store_true",
        help="失敗対象のみ詳細メッセージを標準出力にも表示する（通常はログのみ）。",
    )
    parser.add_argument(
        "--show-all-output",
        action="store_true",
        help="全対象の詳細メッセージを標準出力にも表示する（通常はログのみ）。",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help=f"ログファイルの出力先（既定: {DEFAULT_LOG_FILE_DISPLAY}）。",
    )
    return parser.parse_args()


def dedupe_ports(ports: list[int]) -> list[int]:
    """
    ポート番号を入力順で重複排除する。
    """
    seen: set[int] = set()
    deduped: list[int] = []
    for port in ports:
        if port not in seen:
            seen.add(port)
            deduped.append(port)
    return deduped


def close_logger_handlers(logger: logging.Logger) -> None:
    """
    既存ハンドラを安全にクローズして解除する。
    """
    for handler in list(logger.handlers):
        try:
            handler.flush()
            handler.close()
        finally:
            logger.removeHandler(handler)


def setup_logger(log_file: Path) -> logging.Logger:
    """
    ロガー初期化。

    - 標準出力: INFO 以上（簡潔）
    - ログファイル: DEBUG 以上（詳細）
    """
    normalized_log_file = log_file.expanduser()
    normalized_log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ping_hosts")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    close_logger_handlers(logger)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(normalized_log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def validate_args(args: argparse.Namespace) -> None:
    """
    引数妥当性チェック。
    """
    if args.count < 1:
        raise ValueError("--count は 1 以上を指定してください。")
    if args.timeout < 1:
        raise ValueError("--timeout は 1 以上を指定してください。")
    if args.tcp_timeout <= 0:
        raise ValueError("--tcp-timeout は 0 より大きい値を指定してください。")
    if args.workers < 1:
        raise ValueError("--workers は 1 以上を指定してください。")
    if not args.list_file.is_file():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {args.list_file}")

    args.tcp_port = dedupe_ports(args.tcp_port)
    for port in args.tcp_port:
        if port < 1 or port > 65535:
            raise ValueError(f"--tcp-port は 1..65535 の範囲で指定してください: {port}")

    if args.mode in (MODE_TCP, MODE_BOTH) and not args.tcp_port:
        raise ValueError("--mode tcp/both の場合は --tcp-port を 1 つ以上指定してください。")


def load_targets(path: Path) -> list[str]:
    """
    ターゲット一覧読み込み。

    ルール:
      - 空行は無視
      - `#` 以降はコメント扱い
    """
    targets: list[str] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            target = raw_line.split("#", 1)[0].strip()
            if target:
                targets.append(target)
    return targets


def build_tasks(targets: list[str], mode: str, tcp_ports: list[int]) -> list[CheckTask]:
    """
    実行モードに応じてジョブ一覧を構築する。
    """
    tasks: list[CheckTask] = []

    if mode in (MODE_ICMP, MODE_BOTH):
        for target in targets:
            tasks.append(CheckTask(protocol=MODE_ICMP, target=target))

    if mode in (MODE_TCP, MODE_BOTH):
        for target in targets:
            for port in tcp_ports:
                tasks.append(CheckTask(protocol=MODE_TCP, target=target, port=port))

    return tasks


def build_ping_command(target: str, count: int, timeout: int) -> list[str]:
    """
    ping 実行コマンドを生成する。
    """
    return ["ping", "-n", "-c", str(count), "-W", str(timeout), target]


def build_tcp_command_repr(target: str, port: int, timeout: float) -> str:
    """
    TCP connect の疑似コマンド文字列を生成する。
    """
    return f"socket.connect_ex({target}:{port}, timeout={timeout})"


def resolve_error_message(error_code: int) -> str:
    """
    数値エラーコードを文字列へ変換する。
    """
    try:
        return os.strerror(error_code)
    except ValueError:
        return "unknown error"


def run_icmp_check(task: CheckTask, count: int, timeout: int, capture_output: bool) -> CheckResult:
    """
    ICMP (ping) 実行。
    """
    cmd = build_ping_command(task.target, count, timeout)
    cmd_text = " ".join(cmd)
    started_at = time.monotonic()

    run_kwargs: dict[str, object] = {"check": False}
    if capture_output:
        run_kwargs["stdout"] = subprocess.PIPE
        run_kwargs["stderr"] = subprocess.STDOUT
        run_kwargs["text"] = True
    else:
        run_kwargs["stdout"] = subprocess.DEVNULL
        run_kwargs["stderr"] = subprocess.DEVNULL
        run_kwargs["text"] = False

    try:
        proc = subprocess.run(cmd, **run_kwargs)
        elapsed = time.monotonic() - started_at
        raw_output = proc.stdout if capture_output else ""
        output = raw_output if isinstance(raw_output, str) else ""
        return CheckResult(
            protocol=MODE_ICMP,
            target=task.target,
            port=None,
            returncode=proc.returncode,
            elapsed_seconds=elapsed,
            output=output,
            command=cmd_text,
        )
    except FileNotFoundError:
        elapsed = time.monotonic() - started_at
        return CheckResult(
            protocol=MODE_ICMP,
            target=task.target,
            port=None,
            returncode=RETURNCODE_PING_NOT_FOUND,
            elapsed_seconds=elapsed,
            output="ping コマンドが PATH 上に見つかりません。",
            command=cmd_text,
        )
    except Exception as exc:
        elapsed = time.monotonic() - started_at
        return CheckResult(
            protocol=MODE_ICMP,
            target=task.target,
            port=None,
            returncode=RETURNCODE_UNEXPECTED_ERROR,
            elapsed_seconds=elapsed,
            output=f"予期しないエラー: {exc}",
            command=cmd_text,
        )


def run_tcp_check(task: CheckTask, timeout: float, capture_output: bool) -> CheckResult:
    """
    TCP connect 実行。
    """
    if task.port is None:
        return CheckResult(
            protocol=MODE_TCP,
            target=task.target,
            port=None,
            returncode=RETURNCODE_UNEXPECTED_ERROR,
            elapsed_seconds=0.0,
            output="内部エラー: TCP タスクにポートが設定されていません。",
            command="N/A",
        )

    cmd_text = build_tcp_command_repr(task.target, task.port, timeout)
    started_at = time.monotonic()

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            rc = sock.connect_ex((task.target, task.port))
            elapsed = time.monotonic() - started_at

            if rc == 0:
                output = f"TCP connect success: {task.target}:{task.port}"
            else:
                output = (
                    f"TCP connect failed: errno={rc} "
                    f"({resolve_error_message(rc)})"
                )

            return CheckResult(
                protocol=MODE_TCP,
                target=task.target,
                port=task.port,
                returncode=rc,
                elapsed_seconds=elapsed,
                output=output if capture_output else "",
                command=cmd_text,
            )
    except socket.gaierror as exc:
        elapsed = time.monotonic() - started_at
        error_code = exc.errno if isinstance(exc.errno, int) else RETURNCODE_UNEXPECTED_ERROR
        return CheckResult(
            protocol=MODE_TCP,
            target=task.target,
            port=task.port,
            returncode=error_code,
            elapsed_seconds=elapsed,
            output=f"名前解決失敗: {exc}",
            command=cmd_text,
        )
    except Exception as exc:
        elapsed = time.monotonic() - started_at
        return CheckResult(
            protocol=MODE_TCP,
            target=task.target,
            port=task.port,
            returncode=RETURNCODE_UNEXPECTED_ERROR,
            elapsed_seconds=elapsed,
            output=f"予期しないエラー: {exc}",
            command=cmd_text,
        )


def run_task(task: CheckTask, args: argparse.Namespace, capture_output: bool) -> CheckResult:
    """
    タスク種別に応じた実行関数を呼び分ける。
    """
    if task.protocol == MODE_ICMP:
        return run_icmp_check(
            task=task,
            count=args.count,
            timeout=args.timeout,
            capture_output=capture_output,
        )
    if task.protocol == MODE_TCP:
        return run_tcp_check(
            task=task,
            timeout=args.tcp_timeout,
            capture_output=capture_output,
        )

    return CheckResult(
        protocol=task.protocol,
        target=task.target,
        port=task.port,
        returncode=RETURNCODE_UNEXPECTED_ERROR,
        elapsed_seconds=0.0,
        output=f"未対応プロトコルです: {task.protocol}",
        command="N/A",
    )


def run_checks_parallel(
    tasks: list[CheckTask],
    args: argparse.Namespace,
    capture_output: bool,
) -> list[CheckResult]:
    """
    複数ジョブを並列実行する。
    """
    effective_workers = min(args.workers, len(tasks))
    indexed_results: dict[int, CheckResult] = {}

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_to_index: dict[Future[CheckResult], int] = {}
        for index, task in enumerate(tasks):
            future = executor.submit(run_task, task, args, capture_output)
            future_to_index[future] = index

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            task = tasks[index]
            try:
                indexed_results[index] = future.result()
            except Exception as exc:
                indexed_results[index] = CheckResult(
                    protocol=task.protocol,
                    target=task.target,
                    port=task.port,
                    returncode=RETURNCODE_UNEXPECTED_ERROR,
                    elapsed_seconds=0.0,
                    output=f"ワーカー回収時エラー: {exc}",
                    command="N/A",
                )

    return [indexed_results[i] for i in range(len(tasks))]


def emit_header(
    logger: logging.Logger,
    args: argparse.Namespace,
    target_count: int,
    task_count: int,
) -> None:
    """
    実行条件ヘッダー表示。
    """
    logger.info("=" * 78)
    logger.info("ネットワーク疎通テスト開始")
    logger.info("入力ファイル : %s", args.list_file)
    logger.info("対象件数     : %s", target_count)
    logger.info("実行モード   : %s", args.mode)
    logger.info("並列数       : %s", min(args.workers, task_count))
    logger.info("実行ジョブ数 : %s", task_count)
    logger.info("ログファイル : %s", args.log_file)
    if args.mode in (MODE_ICMP, MODE_BOTH):
        logger.info("ICMP count/timeout : %s / %ss", args.count, args.timeout)
    if args.mode in (MODE_TCP, MODE_BOTH):
        port_text = ",".join(str(port) for port in args.tcp_port)
        logger.info("TCP ports/timeout  : %s / %ss", port_text, args.tcp_timeout)
    logger.info("=" * 78)

    logger.debug(
        "debug_context: python=%s argv=%s cwd=%s",
        sys.version.replace("\n", " "),
        " ".join(sys.argv),
        str(Path.cwd()),
    )


def emit_results(
    logger: logging.Logger,
    results: list[CheckResult],
    show_fail_output: bool,
    show_all_output: bool,
) -> int:
    """
    結果表示と終了コード決定。
    """
    success_count = 0
    failure_by_code: dict[int, int] = {}
    protocol_total: dict[str, int] = {}
    protocol_success: dict[str, int] = {}

    for result in results:
        protocol_total[result.protocol] = protocol_total.get(result.protocol, 0) + 1

        is_success = result.returncode == 0
        if is_success:
            success_count += 1
            protocol_success[result.protocol] = protocol_success.get(result.protocol, 0) + 1
        else:
            failure_by_code[result.returncode] = failure_by_code.get(result.returncode, 0) + 1

        status = "OK" if is_success else "NG"
        logger.info(
            "[%s] %s %s (%.2fs)",
            status,
            result.protocol.upper(),
            result.endpoint(),
            result.elapsed_seconds,
        )
        logger.debug(
            "detail protocol=%s endpoint=%s returncode=%s elapsed=%.2fs command=\"%s\"",
            result.protocol,
            result.endpoint(),
            result.returncode,
            result.elapsed_seconds,
            result.command,
        )

        # 詳細ログは常にファイルへ残す（標準出力には出ない）。
        if result.output:
            for line in result.output.rstrip().splitlines():
                logger.debug("  %s", line)

        # 指定時のみ詳細を標準出力(INFO)へも表示する。
        should_show_output = show_all_output or (show_fail_output and not is_success)
        if should_show_output and result.output:
            for line in result.output.rstrip().splitlines():
                logger.info("  %s", line)

    total = len(results)
    failed = total - success_count
    logger.info("-" * 78)
    logger.info("Summary: total=%s ok=%s ng=%s", total, success_count, failed)

    for protocol in sorted(protocol_total):
        protocol_ok = protocol_success.get(protocol, 0)
        protocol_ng = protocol_total[protocol] - protocol_ok
        logger.info(
            "Summary[%s]: total=%s ok=%s ng=%s",
            protocol.upper(),
            protocol_total[protocol],
            protocol_ok,
            protocol_ng,
        )

    if failure_by_code:
        details = ", ".join(
            f"{code}x{count}" for code, count in sorted(failure_by_code.items())
        )
        logger.info("return codes: %s", details)
    logger.info("-" * 78)

    return EXIT_ALL_SUCCESS if failed == 0 else EXIT_PARTIAL_FAILURE


def main() -> int:
    """
    エントリーポイント。
    """
    args = parse_args()
    logger: logging.Logger | None = None

    try:
        logger = setup_logger(args.log_file)
    except Exception as exc:
        print(f"ERROR: ログ初期化に失敗しました: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    try:
        validate_args(args)
        targets = load_targets(args.list_file)
        if not targets:
            logger.error(
                "ERROR: 入力ファイルに有効なターゲットがありません: %s",
                args.list_file,
            )
            return EXIT_INPUT_ERROR

        tasks = build_tasks(
            targets=targets,
            mode=args.mode,
            tcp_ports=args.tcp_port,
        )
        if not tasks:
            logger.error("ERROR: 実行ジョブが 0 件です。引数を確認してください。")
            return EXIT_INPUT_ERROR

        # ログファイル詳細化のため、常に出力を取得する。
        capture_output = True
        emit_header(
            logger=logger,
            args=args,
            target_count=len(targets),
            task_count=len(tasks),
        )
        results = run_checks_parallel(
            tasks=tasks,
            args=args,
            capture_output=capture_output,
        )
        return emit_results(
            logger=logger,
            results=results,
            show_fail_output=args.show_fail_output,
            show_all_output=args.show_all_output,
        )
    except (ValueError, FileNotFoundError) as exc:
        logger.error("ERROR: %s", exc)
        return EXIT_INPUT_ERROR
    finally:
        if logger is not None:
            close_logger_handlers(logger)


if __name__ == "__main__":
    raise SystemExit(main())
