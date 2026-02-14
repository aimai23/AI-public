# Network Test Tools

## `ping_hosts.py`

ホスト名または IP アドレスを 1 行ずつ記載したファイルを読み込み、疎通テストを実行します。  
実行モードは以下を選択できます。

- `icmp`: ping 疎通
- `tcp`: 指定ポートへの TCP connect 疎通
- `both`: ICMP + TCP の両方

空行と `#` コメント（行末コメント含む）は無視します。

### 実行例

```bash
python3 tools/ping_hosts.py tools/hosts_example.txt
```

```bash
# TCP 22/443 の疎通確認
python3 tools/ping_hosts.py tools/hosts_example.txt -m tcp -p 22 -p 443
```

```bash
# ICMP + TCP の同時確認
python3 tools/ping_hosts.py tools/hosts_example.txt -m both -p 22
```

### 主なオプション

- `-m, --mode`: 実行モード（`icmp` / `tcp` / `both`、既定: `icmp`）
- `-c, --count`: 1 対象あたりの送信回数（既定: `1`）
- `-t, --timeout`: 1 回あたりの応答待ち秒数（既定: `2`）
- `-p, --tcp-port`: TCP 疎通確認ポート（複数指定可）
- `-T, --tcp-timeout`: TCP connect タイムアウト秒（既定: `2.0`）
- `-w, --workers`: 並列ワーカー数（既定: `20`）
- `-f, --show-fail-output`: 失敗対象のみ詳細メッセージを標準出力にも表示（通常はログファイルのみ）
- `-a, --show-all-output`: 全対象の詳細メッセージを標準出力にも表示（通常はログファイルのみ）
- `-l, --log-file`: ログファイル出力先（既定: `tools/ping_hosts.log`）

### 出力

- 標準出力: 実行ヘッダー、対象ごとの OK/NG、Summary
- ログファイル: 標準出力内容に加えて、実行コマンド・返却コード・詳細メッセージを保存

### 終了コード

- `0`: 全対象成功
- `1`: 1 件以上失敗
- `2`: 引数エラー / ファイルエラー / 対象 0 件
