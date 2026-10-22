"""Streamlit 网页：上传订单表 → 自动建模 → 输出该联系哪些客户。

启动：
    streamlit run app.py
"""
import io
import time
import os

import numpy as np
import pandas as pd
import streamlit as st

import autoforecast as af

st.set_page_config(page_title='复购预测 · 自动建模', page_icon='📈', layout='wide')


@st.cache_data(show_spinner=False)
def read_upload(name: str, data: bytes) -> pd.DataFrame:
    ext = os.path.splitext(name)[1].lower()
    buf = io.BytesIO(data)
    if ext in ('.xlsx', '.xlsm', '.xls'):
        xl = pd.ExcelFile(buf)
        frames = [xl.parse(s) for s in xl.sheet_names]
        return pd.concat([f for f in frames if len(f)], ignore_index=True)
    return pd.read_csv(buf, low_memory=False)


def compute(raw, roles, horizon):
    """跑完整流程。结果存进 session_state —— 按钮只在点击的那一次返回 True。"""
    X = af.prepare(raw, roles)
    res, summary, cuts = af.backtest(X, verbose=False)
    if res.empty:
        return {'error': '历史数据太短，切不出回测窗口（至少需要 %d 天以上）。' % (horizon * 3)}
    champion = summary.index[0]
    out, model, Xtr, ytr = af.forecast(X, champion, af.make_models()[champion], horizon=horizon)
    ref_y, ref_s = af.last_window_detail(X, champion, cuts)
    return dict(X=X, res=res, summary=summary, cuts=cuts, out=out, ytr=ytr,
                champion=champion, ref_y=ref_y, ref_s=ref_s)


def main():
    st.title('复购预测 · 自动建模')
    st.caption('上传一份订单明细表，系统自动切时间窗口、自动挑模型、输出该联系的客户名单。'
               '不需要懂机器学习。')

    ss = st.session_state
    ss.setdefault('ana', None)
    ss.setdefault('use_demo', False)

    up = st.file_uploader('上传订单表（Excel 或 CSV）', type=['xlsx', 'xls', 'xlsm', 'csv'])
    demo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'demo_sample.csv')
    if up is None:
        st.info('需要的列：客户编号、下单日期、金额（或 数量×单价）。其它列可有可无。')
        if not os.path.exists(demo_path):
            return
        if st.button('没有数据？用内置示例跑一遍（600 个真实客户，8 万行）'):
            ss.use_demo = True
        if not ss.use_demo:
            return
        raw = pd.read_csv(demo_path, low_memory=False)
        st.success('示例数据：%d 行 × %d 列' % raw.shape)
        src = 'demo'
    else:
        with st.spinner('正在读取文件…'):
            raw = read_upload(up.name, up.getvalue())
        st.success('读取成功：%d 行 × %d 列' % raw.shape)
        src = up.name
    with st.expander('看一眼前 5 行'):
        st.dataframe(raw.head(), use_container_width=True)

    # ---------------- 列角色
    st.subheader('1. 确认列的用途')
    auto = af.detect_columns(raw)
    cols = ['（未选）'] + list(raw.columns)

    if auto.get('customer') and auto.get('date'):
        n_cust = raw[auto['customer']].nunique(dropna=True)
        ratio = n_cust / max(len(raw), 1)
        st.caption('自动识别结果：%d 个客户 / %d 行（平均每个客户 %.1f 行）'
                   % (n_cust, len(raw), len(raw) / max(n_cust, 1)))
        if ratio > 0.5:
            st.warning('客户编号这一列的唯一值接近行数，像是选错了列（可能选成了订单号或商品号）。')
        elif ratio < 0.0005:
            st.warning('客户编号这一列只有 %d 个不同取值，看起来太少，可能选错了列。' % n_cust)

    def sel(label, key, default):
        i = cols.index(default) if default in cols else 0
        return st.selectbox(label, cols, index=i, key=key)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        date_col = sel('下单日期', 'date_col', auto.get('date'))
    with c2:
        cust_col = sel('客户编号', 'cust_col', auto.get('customer'))
    with c3:
        inv_col = sel('订单编号（可选）', 'inv_col', auto.get('invoice'))
    with c4:
        amt_col = sel('金额（单列）', 'amt_col', auto.get('amount'))

    c5, c6, c7 = st.columns(3)
    with c5:
        qty_col = sel('数量（配合单价）', 'qty_col', auto.get('quantity'))
    with c6:
        prc_col = sel('单价（配合数量）', 'prc_col', auto.get('price'))
    with c7:
        horizon = st.number_input('预测未来多少天', 30, 365, 90, step=30)

    if '（未选）' in (date_col, cust_col):
        st.error('至少要有「下单日期」和「客户编号」两列。')
        return
    if amt_col == '（未选）' and (qty_col == '（未选）' or prc_col == '（未选）'):
        st.error('金额缺一个来源：要么选一列金额，要么选「数量 + 单价」两列。')
        return

    roles = {'date': date_col, 'customer': cust_col}
    if inv_col != '（未选）':
        roles['invoice'] = inv_col
    if amt_col != '（未选）':
        roles['amount'] = amt_col
        roles['amount_mode'] = 'single'
    else:
        roles['quantity'], roles['price'] = qty_col, prc_col
        roles['amount_mode'] = 'qty_x_price'

    sig = '|'.join([src, date_col, cust_col, inv_col, amt_col, qty_col, prc_col, str(horizon)])
    ss.setdefault('sig', None)
    if st.button('开始分析', type='primary'):
        with st.spinner('自动切窗口、给每个模型打分、训练、预测…'):
            ss.ana = compute(raw, roles, horizon)
        ss.sig = sig

    if ss.ana is None or ss.sig != sig:
        return
    ana = ss.ana
    if ana.get('error'):
        st.error(ana['error'])
        return

    X, res, summary, cuts = ana['X'], ana['res'], ana['summary'], ana['cuts']
    out, ytr, champion = ana['out'], ana['ytr'], ana['champion']

    st.write('清洗后 %d 行，%d 个客户，时间范围 %s ~ %s'
             % (len(X), X['cust'].nunique(), X['dt'].min().date(), X['dt'].max().date()))

    # ---------------- 回测
    st.subheader('2. 自动切窗口 + 滚动回测')

    left, right = st.columns([3, 2])
    with left:
        st.write('每个时间窗口上的 AUC（越高越好，0.5 = 瞎猜）')
        st.dataframe(res.pivot(index='model', columns='window', values='auc').round(4),
                     use_container_width=True)
    with right:
        st.write('跨窗口平均成绩')
        st.dataframe(summary.round(4), use_container_width=True)

    champion = summary.index[0]
    st.success('自动选出：%s（跨 %d 个窗口平均 AUC %.3f）'
               % (champion, len(res['window'].unique()), summary.loc[champion, 'auc_mean']))

    # ---------------- 预测
    st.subheader('3. 生成名单')
    n_new = int(out['is_new'].sum())
    st.write('训练用 %d 人（其中会复购的 %d 人，占 %.1f%%）｜本次预测覆盖 %d 人'
             % (len(ytr), ytr.sum(), 100 * ytr.mean(), len(out)))
    if n_new:
        st.warning('名单里有 %d 人是在最近 %d 天才第一次出现的客户。模型训练时没见过他们，'
                   '分数不可信，已排在最后并标记 is_new=1，经济测算也把他们剔除了。'
                   % (n_new, horizon))

    # ---------------- 经济价值
    st.subheader('4. 该联系多少人')
    st.caption('系统不替你定。先选你这次的目的，系统给出一个参考长度；滑块仍然可以自己拖，拖到哪数字跟着变。')
    e1, e2 = st.columns(2)
    with e1:
        cost = st.number_input('联系一个客户的成本（元）', 1.0, 10000.0, 50.0, step=10.0)
    with e2:
        value = st.number_input('一次复购带来的毛利（元）', 1.0, 100000.0, 200.0, step=50.0)

    prof, best = af.profit_table(out, ana['ref_y'], ana['ref_s'], cost, value)
    if prof is None or best is None:
        st.warning('回测窗口不足，算不出实测命中曲线，这块先跳过。')
        return

    pool_n = int(prof['N'].max())
    total_hits = float(prof['预计命中'].iloc[-1])

    def at(n):
        """取表里最接近 n 的那一行。"""
        return prof.iloc[(prof['N'] - int(n)).abs().argmin()]

    def pack(r):
        return {'联系人数': int(r['N']),
                '预计命中': round(float(r['预计命中']), 1),
                '命中率': round(float(r['预计命中率']), 4),
                '覆盖会复购客户': round(100.0 * float(r['预计命中']) / total_hits, 1),
                '净收益': round(float(r['净收益']), 0)}

    goal = st.radio('你这次的目的', ['净收益最大', '控制预算', '覆盖一定比例的会复购客户'],
                    horizontal=True, key='goal')
    cover = 80
    budget = min(10000.0, cost * pool_n)

    if goal == '控制预算':
        budget = st.number_input('这次最多愿意花在触达上的钱（元）',
                                 float(cost * 10), float(cost * pool_n),
                                 float(min(10000.0, cost * pool_n)), step=500.0)
        inside = prof[prof['成本'] <= budget]
        ref_n = int(inside['N'].max()) if len(inside) else int(prof['N'].min())
        r = at(ref_n)
        st.info('预算 %.0f 元 → 最多联系 %d 人（花 %.0f 元），预计命中 %.0f 人、命中率 %.1f%%、净收益 %.0f 元。'
                % (budget, ref_n, r['成本'], r['预计命中'], 100 * r['预计命中率'], r['净收益']))
    elif goal == '覆盖一定比例的会复购客户':
        cover = st.slider('想覆盖百分之多少「会复购」的客户', 5, 100, 80, step=5)
        need = total_hits * cover / 100.0
        ok = prof[prof['预计命中'] >= need]
        ref_n = int(ok['N'].min()) if len(ok) else pool_n
        r = at(ref_n)
        st.info('要覆盖 %d%% 的会复购客户（约 %.0f 人），需要联系 %d 人（占客户池 %.0f%%），'
                '这批人的命中率 %.1f%%。'
                % (cover, need, ref_n, 100.0 * ref_n / pool_n, 100 * r['预计命中率']))
    else:
        ref_n = int(best['N'])
        r = at(ref_n)
        st.info('参考：净收益最高点在 N = %d（命中率 %.1f%%，净收益 %.0f 元）。'
                '这只是最高点，不是唯一答案 —— 想留更多客户就往上拖，想省钱就往下拖。'
                % (ref_n, 100 * r['预计命中率'], r['净收益']))

    lo = 10 if pool_n >= 10 else 1
    default_n = int(min(max(lo, ref_n), pool_n))
    n_pick = st.slider('你决定联系多少人', lo, pool_n, default_n, step=1,
                       help='名单越长，命中率越低，但能覆盖更多会回头的人',
                       key='n_pick_%d' % ref_n)
    row = at(n_pick)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric('联系人数', '%d 人' % int(row['N']))
    m2.metric('预计命中', '%.0f 人' % row['预计命中'])
    m3.metric('预计命中率', '%.1f%%' % (100 * row['预计命中率']),
              delta='比随机高 %.1f 个点' % (100 * (row['预计命中率'] - ytr.mean())))
    m4.metric('覆盖会复购客户', '%.1f%%' % (100 * row['预计命中'] / total_hits))
    m5.metric('净收益', '%.0f 元' % row['净收益'])

    if row['净收益'] < 0:
        st.warning('按这个目的和成本，这个名单长度是亏钱的（净收益 %.0f 元）：命中率 %.1f%% 低于门槛 C/V = %.2f。'
                   '要么把名单缩短，要么把单次触达成本压下来。'
                   % (row['净收益'], 100 * row['预计命中率'], cost / value))

    st.write('净收益随名单长度的变化')
    chart = prof.set_index('N')['净收益'].rename('净收益（元）')
    st.line_chart(chart, height=260)

    with st.expander('三种目的放一起看：各自要联系多少人'):
        need = total_hits * cover / 100.0
        ok = prof[prof['预计命中'] >= need]
        c_row = at(int(ok['N'].min())) if len(ok) else at(pool_n)
        ins = prof[prof['成本'] <= budget]
        b_row = at(int(ins['N'].max())) if len(ins) else at(prof['N'].min())
        comp = [dict(目的='净收益最大', **pack(at(best['N']))),
                dict(目的='预算 %.0f 元' % budget, **pack(b_row)),
                dict(目的='覆盖 %d%% 会复购' % cover, **pack(c_row))]
        st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)
        st.caption('三行是同一份预测分数、同一套 C/V，只是「目的」不同。'
                   '要更长的名单就得多花钱，要省预算就得接受漏掉一部分会回头的人。')

    with st.expander('成本不确定？看这张：不同成本下名单会变成什么样'):
        rows = []
        for c in [5, 20, 50, 100, 200]:
            _, b = af.profit_table(out, ana['ref_y'], ana['ref_s'], c, value)
            if b is None:
                continue
            rows.append({'联系成本': c, '门槛 C/V': round(c / value, 3),
                         '最优 N': int(b['N']), '命中率': round(float(b['预计命中率']), 4),
                         '净收益': round(float(b['净收益']), 0)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption('门槛 C/V 的意思是：命中率低于这个数，多联系一个人就亏钱。'
                   '成本越低名单越长，成本高到 150 以上时，只剩打分最高的那几十个人值得打。')

    with st.expander('看完整的经济明细表'):
        st.dataframe(prof.round(4), use_container_width=True)

    topn = st.slider('查看前几名', 10, min(2000, len(out)), min(200, len(out)), step=10)
    st.dataframe(out.head(topn).round(4), use_container_width=True)

    st.download_button('下载完整名单（CSV）',
                       out.to_csv(index=False).encode('utf-8-sig'),
                       file_name='复购预测名单.csv', mime='text/csv')
    st.download_button('下载模型成绩（CSV）',
                       summary.to_csv().encode('utf-8-sig'),
                       file_name='模型成绩.csv', mime='text/csv')


if __name__ == '__main__':
    main()
