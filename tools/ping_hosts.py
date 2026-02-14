#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
ツール名:
  ping_hosts.py

目的:
  ホスト名または IP アドレスの一覧ファイルを読み込み、各対象へ ping を実行する。
  実行結果は「標準出力」と「ログファイル」の両方へ同時出力する。

保守設計:
  1. 設定値/終了コードを定数化し、仕様変更時の修正箇所を最小化する。
  2. 処理を責務ごとに関数分割し、読みやすさと改修しやすさを両立する。
  3. 並列実行しつつ結果表示順は入力順を維持し、運用確認を容易にする。
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
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# 仕様定数
# =============================================================================
# 既定値はここだけを見れば分かるように集約する。
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COUNT = 1
DEFAULT_TIMEOUT = 2
DEFAULT_WORKERS = 20
DEFAULT_LOG_FILE = SCRIPT_DIR / "ping_hosts.log"
DEFAULT_LOG_FILE_DISPLAY = "tools/ping_hosts.log"

# 終了コードを定数化し、呼び出し側（CI/監視）への契約を明示する。
EXIT_ALL_SUCCESS = 0
EXIT_PARTIAL_FAILURE = 1
EXIT_INPUT_ERROR = 2

# 個別ターゲットの実行時に使う疑似エラーコード。
RETURNCODE_PING_NOT_FOUND = 127
RETURNCODE_UNEXPECTED_ERROR = 99


# Python 3.10 未満では dataclass(slots=True) が未対応のため互換分岐する。
DATACLASS_KWARGS = {"slots": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KWARGS)
class PingResult:
    """
    1 ターゲット分の ping 実行結果。

    target:
      対象ホスト名または IP
    returncode:
      ping コマンドの終了コード
    elapsed_seconds:
      実行時間（秒）
    output:
      ping コマンドの出力
    command:
      実際に実行したコマンド文字列（ログ追跡用）
    """

    target: str
    returncode: int
    elapsed_seconds: float
    output: str
    command: str


def parse_args() -> argparse.Namespace:
    """
    CLI 引数定義。

    注意:
      help 文の既定値は定数から組み立て、実装と説明の不一致を防ぐ。
    """
    parser = argparse.ArgumentParser(
        description="リストファイル内のホスト名/IP に対して ping テストを実行します。"
    )
    parser.add_argument(
        "list_file",
        type=Path,
        help="対象一覧ファイル（1 行 1 ホスト名/IP）。",
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"1 対象あたりの ICMP 送信回数（既定: {DEFAULT_COUNT}）。",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"1 回あたりの応答待ち秒数（既定: {DEFAULT_TIMEOUT}）。",
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
        help="失敗対象のみ ping 生出力を標準出力にも表示する（通常はログのみ）。",
    )
    parser.add_argument(
        "--show-all-output",
        action="store_true",
        help="全対象の ping 生出力を標準出力にも表示する（通常はログのみ）。",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help=f"ログファイルの出力先（既定: {DEFAULT_LOG_FILE_DISPLAY}）。",
    )
    return parser.parse_args()


def close_logger_handlers(logger: logging.Logger) -> None:
    """
    既存ハンドラを安全にクローズして解除する。

    同一プロセスで複数回呼ばれるケースでファイルハンドルリークを防ぐ。
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

    要件:
      - 標準出力へ表示
      - ファイルへ保存
    """
    normalized_log_file = log_file.expanduser()
    normalized_log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ping_hosts")
    # ルートレベルは DEBUG にして、ハンドラ側で出力先ごとの粒度を制御する。
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # 既存ハンドラを明示的に閉じてから再設定する。
    close_logger_handlers(logger)

    stream_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    # 標準出力は簡潔表示のみ。
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(normalized_log_file, mode="a", encoding="utf-8")
    # ログファイルは詳細表示。
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
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
    if args.workers < 1:
        raise ValueError("--workers は 1 以上を指定してください。")
    if not args.list_file.is_file():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {args.list_file}")


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
            # 例: "10.0.0.1  # db server" -> "10.0.0.1"
            target = raw_line.split("#", 1)[0].strip()
            if target:
                targets.append(target)
    return targets


def build_ping_command(target: str, count: int, timeout: int) -> list[str]:
    """
    ping 実行コマンドを生成する。

    `-n` を付け、名前解決の逆引きによる遅延を抑制する。
    """
    return ["ping", "-n", "-c", str(count), "-W", str(timeout), target]


def ping_target(
    target: str, count: int, timeout: int, capture_output: bool
) -> PingResult:
    """
    1 ターゲットへ ping 実行。

    capture_output=False の場合は出力を捨て、メモリ使用量を抑える。
    """
    cmd = build_ping_command(target=target, count=count, timeout=timeout)
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
        return PingResult(
            target=target,
            returncode=proc.returncode,
            elapsed_seconds=elapsed,
            output=output,
            command=cmd_text,
        )
    except FileNotFoundError:
        elapsed = time.monotonic() - started_at
        return PingResult(
            target=target,
            returncode=RETURNCODE_PING_NOT_FOUND,
            elapsed_seconds=elapsed,
            output="ping コマンドが PATH 上に見つかりません。",
            command=cmd_text,
        )
    except Exception as exc:
        elapsed = time.monotonic() - started_at
        return PingResult(
            target=target,
            returncode=RETURNCODE_UNEXPECTED_ERROR,
            elapsed_seconds=elapsed,
            output=f"予期しないエラー: {exc}",
            command=cmd_text,
        )


def run_ping_parallel(
    targets: list[str],
    count: int,
    timeout: int,
    workers: int,
    capture_output: bool,
) -> list[PingResult]:
    """
    ping の並列実行。

    実行完了順で回収し、最後に入力順で配列へ戻す。
    """
    effective_workers = min(workers, len(targets))
    indexed_results: dict[int, PingResult] = {}

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_to_index: dict[Future[PingResult], int] = {}
        for index, target in enumerate(targets):
            future = executor.submit(
                ping_target,
                target,
                count,
                timeout,
                capture_output,
            )
            future_to_index[future] = index

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                indexed_results[index] = future.result()
            except Exception as exc:
                # 念のため回収失敗時も結果として残し、全体処理は継続する。
                indexed_results[index] = PingResult(
                    target=targets[index],
                    returncode=RETURNCODE_UNEXPECTED_ERROR,
                    elapsed_seconds=0.0,
                    output=f"ワーカー回収時エラー: {exc}",
                    command="N/A",
                )

    return [indexed_results[i] for i in range(len(targets))]


def emit_header(logger: logging.Logger, args: argparse.Namespace, target_count: int) -> None:
    """
    実行条件ヘッダー表示。
    """
    logger.info("=" * 78)
    logger.info("Ping テストツール開始")
    logger.info("入力ファイル : %s", args.list_file)
    logger.info("対象件数     : %s", target_count)
    logger.info("count/timeout: %s / %ss", args.count, args.timeout)
    logger.info("並列数       : %s", min(args.workers, target_count))
    logger.info("ログファイル : %s", args.log_file)
    logger.info("=" * 78)
    logger.debug(
        "debug_context: python=%s argv=%s cwd=%s",
        sys.version.replace("\n", " "),
        " ".join(sys.argv),
        str(Path.cwd()),
    )


def emit_results(
    logger: logging.Logger,
    results: list[PingResult],
    show_fail_output: bool,
    show_all_output: bool,
) -> int:
    """
    結果表示と終了コード決定。
    """
    success_count = 0
    failure_by_code: dict[int, int] = {}

    for result in results:
        is_success = result.returncode == 0
        if is_success:
            success_count += 1
        else:
            failure_by_code[result.returncode] = failure_by_code.get(result.returncode, 0) + 1

        status = "OK" if is_success else "NG"
        logger.info("[%s] %s (%.2fs)", status, result.target, result.elapsed_seconds)
        logger.debug(
            "detail target=%s returncode=%s elapsed=%.2fs command=\"%s\"",
            result.target,
            result.returncode,
            result.elapsed_seconds,
            result.command,
        )

        # 詳細ログは常にファイルへ残す（標準出力には出ない）。
        if result.output:
            for line in result.output.rstrip().splitlines():
                logger.debug("  %s", line)

        # 互換オプション: 指定時のみ詳細を標準出力(INFO)へも表示する。
        should_show_output = show_all_output or (show_fail_output and not is_success)
        if should_show_output:
            for line in result.output.rstrip().splitlines():
                logger.info("  %s", line)

    total = len(results)
    failed = total - success_count
    logger.info("-" * 78)
    logger.info("Summary: total=%s ok=%s ng=%s", total, success_count, failed)
    if failure_by_code:
        # 例: "return codes: 1x3, 2x1"
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

        # ログファイル詳細化のため、常に出力は取得する。
        capture_output = True
        emit_header(logger, args, target_count=len(targets))
        results = run_ping_parallel(
            targets=targets,
            count=args.count,
            timeout=args.timeout,
            workers=args.workers,
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
            # ハンドラを閉じて明示的に後片付けする。
            close_logger_handlers(logger)


if __name__ == "__main__":
    raise SystemExit(main())
