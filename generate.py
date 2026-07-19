
'''
checkpointを読む
→ configからGPTを再構築
→ parameterを読み込む
→ promptをencode
→ 次tokenを1つsample
→ 入力末尾へ追加
→ 繰り返す
→ 全tokenをdecode
'''

checkpoint