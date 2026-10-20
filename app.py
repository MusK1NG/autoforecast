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


def main():
    st.title('复购预测 · 自动建模')
    st.caption('上传一份订单明细表，系统自动切时间窗口、自动挑模型、输出该联系的客户名单。'
               '不需要懂机器学习。')

    up = st.file_uploader('上传订单表（Excel 或 CSV）', type=['xlsx', 'xls', 'xlsm', 'csv'])
    if up is None:
        st.info('需要的列：客户编号、下单日期、金额（或 数量×单价）。其它列可有可无。')
        return

    with st.spinner('正在读取文件…'):
        raw = read_upload(up.name, up.getvalue())
    st.success('读取成功：%d 行 × %d 列' % raw.shape)
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

    if not st.button('开始分析', type='primary'):
        return

    t0 = time.time()
    X = af.prepare(raw, roles)
    st.write('清洗后 %d 行，%d 个客户，时间范围 %s ~ %s'
             % (len(X), X['cust'].nunique(), X['dt'].min().date(), X['dt'].max().date()))

    # ---------------- 回测
    st.subheader('2. 自动切窗口 + 滚动回测')
    with st.spinner('正在用多个时间窗口给每个模型打分…'):
        res, summary, cuts = af.backtest(X, verbose=False)
    if res.empty:
        st.error('历史数据太短，切不出可用于回测的窗口（至少需要 %d 天以上）。' % (horizon * 3))
        return

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
    with st.spinner('用冠军模型训练并预测…'):
        out, model, Xtr, ytr = af.forecast(X, champion, af.make_models()[champion], horizon=horizon)

    n_new = int(out['is_new'].sum())
    st.write('训练用 %d 人（其中会复购的 %d 人，占 %.1f%%）｜本次预测覆盖 %d 人'
             % (len(ytr), ytr.sum(), 100 * ytr.mean(), len(out)))
    if n_new:
        st.warning('名单里有 %d 人是在最近 %d 天才第一次出现的客户。模型训练时没见过他们，'
                   '分数不可信，已排在最后并标记 is_new=1，经济测算也把他们剔除了。'
                   % (n_new, horizon))

    # ---------------- 经济价值
    st.subheader('4. 该联系多少人')
    st.caption('命中率随名单变长而下降，联系也有成本。填两个数，系统自动算最优名单长度。')
    e1, e2 = st.columns(2)
    with e1:
        cost = st.number_input('联系一个客户的成本（元）', 1.0, 10000.0, 50.0, step=10.0)
    with e2:
        value = st.number_input('一次复购带来的毛利（元）', 1.0, 100000.0, 200.0, step=50.0)

    ref_y, ref_s = af.last_window_detail(X, champion, cuts)
    prof, best = af.profit_table(out, ref_y, ref_s, cost, value)

    if prof is None:
        st.warning('回测窗口不足，算不出实测命中曲线，这块先跳过。')
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('建议联系人数', '%d 人' % int(best['N']))
        m2.metric('预计命中', '%.0f 人' % best['预计命中'])
        m3.metric('预计命中率', '%.1f%%' % (100 * best['预计命中率']))
        m4.metric('预计净收益', '%.0f 元' % best['净收益'])
        st.caption('命中率来自实测：用最后一个回测窗口上真实发生的结果，'
                   '按名单长度的相对分位迁移到本次名单。不是拍的。')
        if best['净收益'] <= 0:
            st.warning('按这组成本算，投出去是亏的：联系成本高于预期收益。'
                       '要么降低触达成本，要么提高复购价值。')
        with st.expander('看完整的经济曲线'):
            st.dataframe(prof.round(4), use_container_width=True)

    topn = st.slider('查看前几名', 10, min(2000, len(out)), min(200, len(out)), step=10)
    st.dataframe(out.head(topn).round(4), use_container_width=True)

    st.download_button('下载完整名单（CSV）',
                       out.to_csv(index=False).encode('utf-8-sig'),
                       file_name='复购预测名单.csv', mime='text/csv')
    st.download_button('下载模型成绩（CSV）',
                       summary.to_csv().encode('utf-8-sig'),
                       file_name='模型成绩.csv', mime='text/csv')
    st.caption('总耗时 %.1f 秒' % (time.time() - t0))


if __name__ == '__main__':
    main()
