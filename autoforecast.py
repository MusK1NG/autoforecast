"""AutoForecast — 上传一份订单表，自动切窗口、自动选模型、输出预测名单。

设计原则：
1. 用户只需要上传 Excel/CSV，不用懂任何机器学习；
2. 列角色自动识别，识别结果打印出来供人工确认（网页上就是下拉框）；
3. 模型不用 CV 选（已实测会选错），用滚动回测选；
4. 输出不只是一个分数，是一份可执行名单。
"""
import re
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

H = 90  # 标签窗口天数


# ---------------------------------------------------------------- 1. 列角色识别
CUST_HINT = r'(customer|client|user|account|member|buyer|客户|用户|会员)'
ORDER_HINT = r'(invoice|order|transaction|receipt|ticket|单号|订单|发票)'
DATE_HINT = r'(date|time|day|日期|时间)'
AMT_HINT = r'(amount|total|price|sales|revenue|金额|价|收入)'
QTY_HINT = r'(qty|quantity|数量|件数)'

BAD_CUST = r'(stock|sku|code|product|item|商品|编码)'


def _to_dt(s: pd.Series) -> pd.Series:
    """先走快速路径（规整格式），失败率高才用 mixed 逐行推断。"""
    out = pd.to_datetime(s, errors='coerce')
    if out.isna().mean() > 0.2:
        out = pd.to_datetime(s, errors='coerce', format='mixed')
    return out


def _maybe_date(s: pd.Series) -> float:
    if pd.api.types.is_datetime64_any_dtype(s):
        return 1.0
    if s.dtype == object:
        try:
            return float(_to_dt(s).notna().mean())
        except Exception:
            return 0.0
    return 0.0


def detect_columns(df: pd.DataFrame) -> dict:
    """返回 {date, customer, amount, invoice, quantity, price}，识别不出的为 None。

    只在前 5000 行上判断：全量 106 万行做日期解析要几分钟，识别用不着那么多。
    """
    smp = df.head(5000) if len(df) > 5000 else df
    n = len(smp)
    roles = {}
    cols = list(df.columns)

    # 日期列
    cands = [(c, _maybe_date(smp[c])) for c in cols]
    cands = [c for c in cands if c[1] >= 0.95]
    if cands:
        named = [c for c, _ in cands if re.search(DATE_HINT, str(c), re.I)]
        roles['date'] = (named or [cands[0][0]])[0]

    # 订单号
    named = [c for c in cols if re.search(ORDER_HINT, str(c), re.I)]
    if named:
        roles['invoice'] = named[0]

    # 客户列：名字优先，其次统计特征
    named = [c for c in cols
             if re.search(CUST_HINT, str(c), re.I) and not re.search(BAD_CUST, str(c), re.I)]
    if named:
        # 多个命中时取唯一值最少的（客户池通常最小）
        named.sort(key=lambda c: smp[c].nunique(dropna=True))
        roles['customer'] = named[0]
    else:
        pool = []
        for c in cols:
            if c == roles.get('date') or c == roles.get('invoice'):
                continue
            nu = smp[c].nunique(dropna=True)
            if 10 <= nu <= 0.5 * n and nu / n >= 1e-4:
                pool.append((c, nu))
        if pool:
            pool.sort(key=lambda t: abs(np.log(t[1]) - np.log(max(10, n ** 0.5))))
            roles['customer'] = pool[0][0]

    # 金额：单列金额，或者 数量 × 单价
    named = [c for c in cols if re.search(AMT_HINT, str(c), re.I)]
    qty = [c for c in cols if re.search(QTY_HINT, str(c), re.I)]
    prc = [c for c in cols if re.search(r'(price|单价)', str(c), re.I)]
    if qty and prc:
        roles['quantity'], roles['price'] = qty[0], prc[0]
        roles['amount_mode'] = 'qty_x_price'
    elif named:
        # 排除数量列
        named = [c for c in named if c not in qty]
        if named:
            roles['amount'] = named[0]
            roles['amount_mode'] = 'single'
    return roles


def prepare(df: pd.DataFrame, roles: dict) -> pd.DataFrame:
    """统一成 cust / dt / amount / invoice 四列。"""
    out = pd.DataFrame()
    out['cust'] = df[roles['customer']]
    out['dt'] = _to_dt(df[roles['date']])
    inv = df[roles['invoice']] if roles.get('invoice') else out['dt'].astype(str)
    out['invoice'] = inv.astype(str)

    if roles.get('amount_mode') == 'qty_x_price':
        out['amount'] = pd.to_numeric(df[roles['quantity']], errors='coerce') * \
                        pd.to_numeric(df[roles['price']], errors='coerce')
        out['qty'] = pd.to_numeric(df[roles['quantity']], errors='coerce')
    else:
        out['amount'] = pd.to_numeric(df[roles['amount']], errors='coerce')
        out['qty'] = np.nan

    out = out.dropna(subset=['cust', 'dt'])
    # 商品列（可选），用来算"买过几种东西"
    stock_col = next((c for c in df.columns
                      if re.search(r'(stock|sku|product|item|商品|编码)', str(c), re.I)), None)
    out['stock'] = df.loc[out.index, stock_col].astype(str) if stock_col else np.nan
    # 清洗：取消单、退货、零金额
    out = out[~out['invoice'].str.upper().str.startswith('C')]
    out = out[(out['amount'] > 0) & (out['qty'].isna() | (out['qty'] > 0))]
    out['dt'] = out['dt'].dt.normalize()
    # 客户号：数值型且全是整数时，去掉 .0，否则编号会显示成 17841.0
    s = out['cust']
    if pd.api.types.is_numeric_dtype(s):
        out['cust'] = s.apply(lambda v: str(int(v)) if float(v).is_integer() else str(v))
    else:
        out['cust'] = s.astype(str)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- 2. 特征表
def build(X: pd.DataFrame, cutoff: pd.Timestamp, horizon: int = H):
    """cutoff 之前的信息做特征，cutoff 之后 horizon 天做标签。"""
    feat = X[X['dt'] <= cutoff]
    lab = X[(X['dt'] > cutoff) & (X['dt'] <= cutoff + pd.Timedelta(days=horizon))]
    if feat.empty:
        return None, None
    g = feat.groupby('cust').agg(first=('dt', 'min'), last=('dt', 'max'),
                                 freq=('invoice', 'nunique'),
                                 mon=('amount', 'sum'))
    F = pd.DataFrame(index=g.index)
    F['recency'] = (cutoff - g['last']).dt.days
    F['tenure'] = (cutoff - g['first']).dt.days
    F['log_frequency'] = np.log1p(g['freq'])
    F['log_monetary'] = np.log1p(g['mon'])
    # 每个客户买过多少种商品（有商品列时用得到）
    if 'stock' in feat.columns:
        n_item = feat.groupby('cust')['stock'].nunique()
        F['log_items'] = np.log1p(n_item.reindex(F.index).fillna(0))
    F['log_avg'] = np.log1p(g['mon'] / g['freq'].clip(lower=1))
    buyers = set(lab['cust'])
    y = pd.Series([1 if c in buyers else 0 for c in F.index], index=F.index)
    return F, y


# ---------------------------------------------------------------- 3. 模型库
def make_models():
    return {
        'LogisticRegression': lambda: Pipeline([('sc', StandardScaler()),
                                                ('m', LogisticRegression(C=1, max_iter=2000))]),
        'Ridge': lambda: Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=10))]),
        'RandomForest': lambda: RandomForestClassifier(max_depth=5, n_estimators=200,
                                                       random_state=0, n_jobs=-1),
    }


def scores(m, X):
    return m.predict_proba(X)[:, 1] if hasattr(m, 'predict_proba') else m.predict(X)


# ---------------------------------------------------------------- 4. 滚动回测
def cutoffs_for(X: pd.DataFrame, horizon: int = H, n_windows: int = 4):
    end = X['dt'].max()
    last = end - pd.Timedelta(days=horizon)          # 最后一个可切的位置
    step = pd.Timedelta(days=max(30, horizon // 2))
    cuts = [last - step * k for k in range(n_windows - 1, -1, -1)]
    start = X['dt'].min() + pd.Timedelta(days=max(horizon * 2, 120))
    return [c for c in cuts if c >= start]


def backtest(X, verbose=True):
    cuts = cutoffs_for(X)
    tabs = {c: build(X, c) for c in cuts}
    cuts = [c for c in cuts if tabs[c][0] is not None]
    models = make_models()
    table = []
    for i in range(1, len(cuts)):
        ctr, cte = cuts[i - 1], cuts[i]
        Xtr, ytr = tabs[ctr]
        Xte, yte = tabs[cte]
        if yte.nunique() < 2:
            continue
        cols = [c for c in Xtr.columns if c in Xte.columns]
        for name, f in models.items():
            m = f()
            m.fit(Xtr[cols], ytr)
            s = scores(m, Xte[cols])
            table.append({'window': str(cte.date()), 'model': name,
                          'auc': roc_auc_score(yte, s),
                          'pr_auc': average_precision_score(yte, s),
                          'acc': accuracy_score(yte, (s > 0.5).astype(int)),
                          'n': len(yte), 'pos': int(yte.sum())})
    res = pd.DataFrame(table)
    if verbose and not res.empty:
        print('\n滚动回测（每次用过去训练、用接下来的 %d 天考试）' % H)
        pv = res.pivot(index='model', columns='window', values='auc')
        print(pv.round(6).to_string())
    summary = res.groupby('model').agg(auc_mean=('auc', 'mean'), auc_min=('auc', 'min'),
                                       auc_std=('auc', 'std'),
                                       pr_auc_mean=('pr_auc', 'mean')).sort_values('auc_mean', ascending=False)
    if verbose and not summary.empty:
        print('\n跨窗口成绩（按平均 AUC 排）')
        print(summary.round(6).to_string())
    return res, summary, cuts


def last_window_detail(X, model_name, cuts):
    """取最后一个回测窗口上冠军模型的 (真实标签, 分数)，用来画实测命中曲线。"""
    if len(cuts) < 2:
        return None, None
    Xtr, ytr = build(X, cuts[-2])
    Xte, yte = build(X, cuts[-1])
    if Xtr is None or Xte is None or yte.nunique() < 2:
        return None, None
    cols = [c for c in Xtr.columns if c in Xte.columns]
    m = make_models()[model_name]()
    m.fit(Xtr[cols], ytr)
    return yte.values, scores(m, Xte[cols])


def hit_curve(y_true, s, n_points=60):
    """实测的 top-N 命中曲线：按分数从高到低切，每个长度上的真实命中率。"""
    o = np.argsort(-np.asarray(s))
    y = np.asarray(y_true)[o]
    cum = np.cumsum(y)
    n = np.arange(1, len(y) + 1)
    idx = np.unique(np.linspace(0, len(y) - 1, min(n_points, len(y))).astype(int))
    frac = n[idx] / len(y)
    return frac, cum[idx] / n[idx], cum[idx]


def profit_table(out, ref_y, ref_s, cost, value, n_grid=300):
    """用实测命中曲线按分位迁移到本次名单，算每个名单长度 N 的净收益。

    out 里的 is_new（新客）不参与：模型没在训练里见过他们。
    """
    pool = out[~out['is_new']] if 'is_new' in out.columns else out
    if ref_y is None or len(pool) == 0:
        return None, None
    frac_ref, rate_ref, _ = hit_curve(ref_y, ref_s)
    Ns = np.unique(np.linspace(10, len(pool), n_grid).astype(int))
    rate = np.interp(Ns / len(pool), frac_ref, rate_ref)   # 同分位对齐
    hits = rate * Ns
    tab = pd.DataFrame({'N': Ns, '预计命中率': rate, '预计命中': hits,
                        '成本': Ns * cost, '毛收益': hits * value,
                        '净收益': hits * value - Ns * cost})
    return tab, tab.loc[tab['净收益'].idxmax()]


# ---------------------------------------------------------------- 5. 最终预测
def forecast(X, model_name, champion_factory, horizon=H):
    end = X['dt'].max()
    c_train = end - pd.Timedelta(days=horizon)
    Xtr, ytr = build(X, c_train)
    m = champion_factory()
    m.fit(Xtr, ytr)
    Xpred, _ = build(X, end)           # 用全部数据，预测下一个 horizon
    cols = list(Xtr.columns)
    Xpred = Xpred.reindex(columns=cols)
    if Xpred.isna().any().any():
        Xpred = Xpred.fillna(0)
    s = scores(m, Xpred)
    out = pd.DataFrame({'cust': Xpred.index, 'score': s})
    # 训练时没见过的客户（最后 90 天才首次出现）：模型没学过他们，分数不可信
    out['is_new'] = ~out['cust'].isin(Xtr.index)
    out = out.sort_values(['is_new', 'score'], ascending=[True, False]).reset_index(drop=True)
    # 附上业务列，方便直接拿去用
    last = X.groupby('cust').agg(last_order=('dt', 'max'), n_orders=('invoice', 'nunique'),
                                 total_amount=('amount', 'sum'))
    out = out.merge(last.reset_index(), on='cust', how='left')
    out['last_order'] = out['last_order'].dt.date
    return out.reset_index(drop=True), m, Xtr, ytr
