#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
ツール名:
  hulft_send.py

目的:
  HULFT配信ファイルID(およびファイル名)の一覧ファイルを読み込み、utlsend コマンドを連続実行する。
  
  実行結果は「標準出力」と「ログファイル」の両方へ同時出力する。
  標準出力は簡潔表示、ログファイルは詳細表示とする。

仕様:
  1. 入力リスト形式: 「FILE_ID [FILE_PATH]」 (スペース区切り)
  2. コマンド構成:
     - FILE_ID のみ: `utlsend -f FILE_ID`
     - FILE_PATH あり: `utlsend -f FILE_ID -file FILE_PATH`
     (※ -file オプションは HULFT の動的パラメータ指定が有効な場合のみ機能します)

保守設計:
  1. 設定値/終了コードを定数化し、仕様変更時の修正箇所を最小化する。
  2. 結果表示順は入力順を維持し、運用確認を容易にする。
  3. エラーは可能な限り結果として集約し、途中停止を避ける。
  4. --dry-run モードを実装し、実行コマンドを事前に確認可能にする。

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
from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# 仕様定数
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TIMEOUT = 60 # HULFT 転送待ちタイムアウト(秒)
DEFAULT_LOG_FILE = SCRIPT_DIR / "hulft_send.log"
DEFAULT_LOG_FILE_DISPLAY = "tools/hulft_send/hulft_send.log"

# HULFT コマンド
CMD_UTLSEND = "utlsend"

# 終了コード
EXIT_ALL_SUCCESS = 0
EXIT_PARTIAL_FAILURE = 1
EXIT_INPUT_ERROR = 2

# 個別ターゲット実行時の疑似エラーコード
RETURNCODE_CMD_NOT_FOUND = 127
RETURNCODE_UNEXPECTED_ERROR = 99
RETURNCODE_TIMEOUT = 124

# Python 3.10 未満では dataclass(slots=True) が未対応のため互換分岐する。
DATACLASS_KWARGS = {"slots": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KWARGS)
class SendTask:
    """
    実行ジョブ定義。
    
    file_id:
      HULFT 配信ファイルID
    file_path:
      (任意) 送信ファイルパス。指定時は -file オプションで上書きする想定。
    """
    file_id: str
    file_path: str | None = None


@dataclass(**DATACLASS_KWARGS)
class SendResult:
    """
    1 ジョブ分の実行結果。
    
    file_id:
      HULFT 配信ファイルID
    returncode:
      実行結果コード（0 は成功）
    elapsed_seconds:
      実行時間（秒）
    output:
      実行時の詳細メッセージ
    command:
      実行コマンド文字列（ログ追跡用）
    """
    file_id: str
    returncode: int
    elapsed_seconds: float
    output: str
    command: str

    def endpoint(self) -> str:
        """表示用ID"""
        return self.file_id


def parse_args() -> argparse.Namespace:
    """
    CLI 引数定義。
    """
    parser = argparse.ArgumentParser(
        description="リストファイル内の配信IDに対して utlsend を連続実行します。"
    )
    parser.add_argument(
        "list_file",
        type=Path,
        help="対象一覧ファイル（形式: FILE_ID [FILE_PATH]）。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="コマンドを実行せず、生成されるコマンドを表示します。",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"utlsend 実行タイムアウト秒（既定: {DEFAULT_TIMEOUT}）。",
    )
    parser.add_argument(
        "-f",
        "--show-fail-output",
        action="store_true",
        help="失敗対象のみ詳細メッセージを標準出力にも表示する（通常はログのみ）。",
    )
    parser.add_argument(
        "-a",
        "--show-all-output",
        action="store_true",
        help="全対象の詳細メッセージを標準出力にも表示する（通常はログのみ）。",
    )
    parser.add_argument(
        "-l",
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help=f"ログファイルの出力先（既定: {DEFAULT_LOG_FILE_DISPLAY}）。",
    )
    return parser.parse_args()


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
    """
    normalized_log_file = log_file.expanduser()
    normalized_log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("hulft_send")
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
    if args.timeout < 1:
        raise ValueError("--timeout は 1 以上を指定してください。")
    if not args.list_file.is_file():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {args.list_file}")


def load_targets(path: Path) -> list[SendTask]:
    """
    ターゲット一覧読み込み。
    
    形式:
      FILE_ID [FILE_PATH]
      # コメント
    """
    tasks: list[SendTask] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for _, raw_line in enumerate(file_obj, 1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            
            parts = line.split()
            file_id = parts[0]
            file_path = parts[1] if len(parts) > 1 else None
            
            tasks.append(SendTask(file_id=file_id, file_path=file_path))
            
    return tasks


def build_command(task: SendTask) -> list[str]:
    """
    utlsend コマンドを生成する。
    """
    cmd = [CMD_UTLSEND, "-f", task.file_id]
    if task.file_path:
        # ファイルパス指定がある場合は -file オプションを追加
        # (HULFT の動的指定設定が必要)
        cmd.extend(["-file", task.file_path])
    return cmd


def run_send(task: SendTask, timeout: int, dry_run: bool) -> SendResult:
    """
    utlsend 実行。
    """
    cmd = build_command(task)
    cmd_text = " ".join(cmd)
    started_at = time.monotonic()

    if dry_run:
        # Dry-run モード
        return SendResult(
            file_id=task.file_id,
            returncode=0,
            elapsed_seconds=0.0,
            output="(dry-run) command would be executed.",
            command=cmd_text,
        )

    run_kwargs: dict[str, object] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,  # Python 3.7+
        "check": False, 
    }

    try:
        proc = subprocess.run(cmd, timeout=timeout, **run_kwargs)
        elapsed = time.monotonic() - started_at
        output = proc.stdout if isinstance(proc.stdout, str) else ""
        return SendResult(
            file_id=task.file_id,
            returncode=proc.returncode,
            elapsed_seconds=elapsed,
            output=output,
            command=cmd_text,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - started_at
        return SendResult(
            file_id=task.file_id,
            returncode=RETURNCODE_TIMEOUT,
            elapsed_seconds=elapsed,
            output=f"タイムアウト ({timeout}秒) しました。",
            command=cmd_text,
        )
    except FileNotFoundError:
        elapsed = time.monotonic() - started_at
        return SendResult(
            file_id=task.file_id,
            returncode=RETURNCODE_CMD_NOT_FOUND,
            elapsed_seconds=elapsed,
            output=f"コマンドが見つかりません: {CMD_UTLSEND}",
            command=cmd_text,
        )
    except Exception as exc:
        elapsed = time.monotonic() - started_at
        return SendResult(
            file_id=task.file_id,
            returncode=RETURNCODE_UNEXPECTED_ERROR,
            elapsed_seconds=elapsed,
            output=f"予期しないエラー: {exc}",
            command=cmd_text,
        )


def run_sequential(
    tasks: list[SendTask],
    args: argparse.Namespace,
) -> list[SendResult]:
    """
    複数ジョブを順次実行する。
    """
    results: list[SendResult] = []

    for task in tasks:
        result = run_send(task, args.timeout, args.dry_run)
        results.append(result)

    return results


def emit_header(
    logger: logging.Logger,
    args: argparse.Namespace,
    task_count: int,
) -> None:
    """
    実行条件ヘッダー表示。
    """
    logger.info("=" * 78)
    logger.info("HULFT 配信ツール開始")
    logger.info("入力ファイル : %s", args.list_file)
    logger.info("実行ジョブ数 : %s", task_count)
    logger.info("ログファイル : %s", args.log_file)
    logger.info("ドライラン   : %s", "ON" if args.dry_run else "OFF")
    logger.info("=" * 78)

    logger.debug(
        "debug_context: python=%s argv=%s cwd=%s",
        sys.version.replace("\n", " "),
        " ".join(sys.argv),
        str(Path.cwd()),
    )


def emit_results(
    logger: logging.Logger,
    results: list[SendResult],
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
        logger.info(
            "[%s] %s (%.2fs)",
            status,
            result.file_id,
            result.elapsed_seconds,
        )
        # 実行コマンドもログ出力（デバッグ用）
        logger.debug(
            "detail file_id=%s returncode=%s elapsed=%.2fs command=\"%s\"",
            result.file_id,
            result.returncode,
            result.elapsed_seconds,
            result.command,
        )

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

    if failure_by_code:
        details = ", ".join(
            f"code{code}x{count}" for code, count in sorted(failure_by_code.items())
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
        tasks = load_targets(args.list_file)
        if not tasks:
            logger.error("ERROR: 実行ジョブが 0 件です。入力ファイルを確認してください。")
            return EXIT_INPUT_ERROR

        emit_header(logger, args, len(tasks))
        results = run_sequential(tasks, args)
        return emit_results(logger, results, args.show_fail_output, args.show_all_output)

    except (ValueError, FileNotFoundError) as exc:
        if logger:
            logger.error("ERROR: %s", exc)
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    finally:
        if logger:
            close_logger_handlers(logger)


if __name__ == "__main__":
    raise SystemExit(main())
