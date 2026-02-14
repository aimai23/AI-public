# Network Test Tools

## `ping_hosts.py`

ホスト名または IP アドレスを 1 行ずつ記載したファイルを読み込み、`ping` テストを実行します。  
空行と `#` コメント（行末コメント含む）は無視します。

### 実行例

```bash
python3 tools/ping_hosts.py tools/hosts_example.txt
```

### 主なオプション

- `-c, --count`: 1 対象あたりの送信回数（既定: `1`）
- `-t, --timeout`: 1 回あたりの応答待ち秒数（既定: `2`）
- `-w, --workers`: 並列ワーカー数（既定: `20`）
- `--show-fail-output`: 失敗対象のみ ping 生出力を標準出力にも表示（通常はログファイルのみ）
- `--show-all-output`: 全対象の ping 生出力を標準出力にも表示（通常はログファイルのみ）
- `--log-file`: ログファイル出力先（既定: `tools/ping_hosts.log`）

### 出力

- 標準出力: 実行ヘッダー、対象ごとの OK/NG、Summary
- ログファイル: 標準出力内容に加えて、実行コマンド・返却コード・ping 生出力を詳細保存

### 終了コード

- `0`: 全対象成功
- `1`: 1 件以上失敗
- `2`: 引数エラー / ファイルエラー / 対象 0 件
