"""命令行入口：传一个 Excel/CSV，自动跑完整流程，输出名单。

用法：
    python run.py "C:/path/to/orders.xlsx" [--horizon 90] [--top 500]
"""
import os
import sys
import time
import hashlib
import argparse

import pandas as pd

import autoforecast as af


def load_any(path, sheet=None):
    """读任意 Excel/CSV。多 sheet 的 Excel 合并成一个表。大文件走缓存。"""
    tag = hashlib.md5((os.path.abspath(path) + str(os.path.getmtime(path))).encode()).hexdigest()[:10]
    cache = os.path.join(os.path.dirname(os.path.abspath(path)), f'.cache_{tag}.pkl')
    if os.path.exists(cache):
        return pd.read_pickle(cache)

    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xlsm', '.xls'):
        xl = pd.ExcelFile(path)
        sheets = [sheet] if sheet else xl.sheet_names
        frames = []
        for s in sheets:
            d = xl.parse(s)
            if len(d):
                frames.append(d)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path, low_memory=False)
    df.to_pickle(cache)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('file')
    ap.add_argument('--sheet', default=None)
    ap.add_argument('--horizon', type=int, default=90)
    ap.add_argument('--top', type=int, default=500)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    t0 = time.time()
    print('读取文件：', a.file)
    raw = load_any(a.file, a.sheet)
    print('  行数 %d，列 %d' % raw.shape)
    print('  列名：', list(raw.columns))

    print('\n[1/4] 自动识别列的用途')
    roles = af.detect_columns(raw)
    for k in ['date', 'customer', 'invoice', 'amount', 'quantity', 'price', 'amount_mode']:
        if k in roles:
            print('  %-12s → %s' % (k, roles[k]))

    missing = [k for k in ('date', 'customer') if k not in roles]
    if missing:
        print('\n识别不出这些关键列，请在网页/命令行里手动指定：', missing)
        sys.exit(1)

    X = af.prepare(raw, roles)
    print('  清洗后 %d 行，%d 个客户，时间 %s ~ %s'
          % (len(X), X['cust'].nunique(), X['dt'].min().date(), X['dt'].max().date()))

    print('\n[2/4] 自动切时间窗口 + 滚动回测选模型')
    res, summary, cuts = af.backtest(X)
    champion = summary.index[0]
    print('\n  → 冠军模型：%s（平均 AUC %.6f）' % (champion, summary.loc[champion, 'auc_mean']))

    print('\n[3/4] 用冠军模型训练 + 预测')
    out, model, Xtr, ytr = af.forecast(X, champion, af.make_models()[champion], horizon=a.horizon)
    print('  训练样本 %d 人（其中复购 %d 人，%.2f%%）'
          % (len(ytr), ytr.sum(), 100 * ytr.mean()))
    print('  预测覆盖 %d 人' % len(out))

    print('\n[4/4] 输出名单 top %d' % a.top)
    top = out.head(a.top).copy()
    print(top.head(10).to_string(index=False))

    dest = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output',
                                 'forecast_list.csv')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    out.to_csv(dest, index=False, encoding='utf-8-sig')
    summary.to_csv(dest.replace('.csv', '_model_scores.csv'), encoding='utf-8-sig')
    print('\n完整名单 →', dest)
    print('模型成绩 →', dest.replace('.csv', '_model_scores.csv'))
    print('耗时 %.1f 秒' % (time.time() - t0))


if __name__ == '__main__':
    main()
