# HULFT 配信ツール (hulft_send)

`utlsend` コマンドを使用して、リストファイルに記載された複数のファイルIDに対して連続配信を行うツールです。
直列実行による確実な処理と、ログ出力機能を備えています。

## 必要要件

- Python 3.7 以上
- HULFT がインストールされており、`utlsend` コマンドにパスが通っていること (RHEL想定)

## ファイル構成

- `hulft_send.py`: ツール本体
- `file_list_example.txt`: 配信リストのサンプル
- `hulft_send.log`: 実行ログ (実行時に生成)

## 使い方

```bash
# 基本的な実行 (リストファイルを指定)
python3 hulft_send.py list.txt

# ドライラン (コマンドを確認するだけで実行しない)
python3 hulft_send.py list.txt --dry-run

# エラー時のみ標準出力にも詳細を表示
python3 hulft_send.py list.txt -f

# 全対象の詳細を標準出力にも表示
python3 hulft_send.py list.txt -a
```

## オプション一覧

| オプション | 説明 | 既定値 |
|---|---|---|
| `list_file` | 対象一覧ファイル（必須） | ― |
| `--dry-run` | コマンドを実行せず表示のみ | OFF |
| `-t, --timeout` | utlsend 実行タイムアウト秒 | 60 |
| `-f, --show-fail-output` | 失敗時のみ詳細を標準出力にも表示 | OFF |
| `-a, --show-all-output` | 全対象の詳細を標準出力にも表示 | OFF |
| `-l, --log-file` | ログファイルの出力先 | `hulft_send.log` |

## リストファイルの形式

1行につき1つの配信IDを記述します。
スペース区切りでファイルパスを指定すると、`-file` オプションとして付与されます（動的ファイル名指定）。

```text
# IDのみ (utlsend -f ID)
FILE_ID_001

# IDとファイル名 (utlsend -f ID -file /path/to/file)
FILE_ID_002 /path/to/override.csv
```

## ログ

- 標準出力には `[OK]` `[NG]` などの概要が表示されます。
- 詳細なログ（実行コマンド、標準出力、エラー出力）は `hulft_send.log` に出力されます。

## 終了コード

| コード | 意味 |
|---|---|
| 0 | 全対象成功 |
| 1 | 1件以上失敗 |
| 2 | 引数不正 / 入力ファイル不正 / 対象0件 / ログ初期化失敗 |
