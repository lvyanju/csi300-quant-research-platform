"""2015—2025 年 DMA 指数择时实验在研究管理台中的只读展示。"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

EXPERIMENT = Path(__file__).resolve().parents[1] / 'research' / 'dma_2015_2025'
SOURCE_URL = ('https://github.com/1376524890/bigdata_fin/blob/master/'
              'real_data/01_%E6%8C%87%E6%95%B0%E6%95%B0%E6%8D%AE/hs300_daily.csv')


@st.cache_data
def _csv(path: str, modified: float) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_csv(path: Path) -> pd.DataFrame:
    return _csv(str(path), path.stat().st_mtime)


def _download(path: Path, label: str) -> None:
    st.download_button(label, path.read_bytes(), file_name=path.name,
                       mime='text/csv' if path.suffix == '.csv' else 'application/octet-stream')


def render_dma() -> None:
    st.subheader('DMA 择时方法复现')
    st.caption('沪深300指数日线｜2015—2025 年｜DMA(10,50,10)｜独立历史实验')
    st.info('沪深300指数不能直接交易。以下交易账本使用可分割的指数单位作价格代理，'
            '与平台其他页签的 510300 ETF 回测口径不同，收益不能直接比较。')
    if st.button('重新运行 DMA 回测', key='dma_replay',
                 disabled=os.environ.get('QUANT_PUBLIC_READ_ONLY') == '1'):
        try:
            with st.spinner('校验历史行情，重算指标、逐日信号与模拟账本…'):
                done = subprocess.run([sys.executable, str(EXPERIMENT / 'run_dma.py')],
                                      cwd=EXPERIMENT, capture_output=True, text=True,
                                      timeout=120, check=True)
            _csv.clear()
            st.success('DMA 回测已重新生成；数据与参数见下方报告。')
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            detail = getattr(exc, 'stderr', '') or str(exc)
            st.error(f'回测未完成：{detail[-1000:]}')
    summary_path = EXPERIMENT / 'outputs' / 'summary.json'
    source_path = EXPERIMENT / 'data' / 'hs300_daily_source.csv'
    if not summary_path.exists() or not source_path.exists():
        st.warning('缺少已保存的 DMA 实验数据或结果，请检查 research/dma_2015_2025。')
        return
    report = json.loads(summary_path.read_text(encoding='utf-8'))
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if digest != report.get('source_sha256'):
        st.error('原始行情与回测记录的数据哈希不一致，请先核查文件，暂不展示历史成绩。')
        return
    if report.get('index_code') != 'sh000300' or report.get('start') != '2015-01-05' or report.get('end') != '2025-12-31':
        st.error('回测记录的标的或日期范围不符合本实验口径。')
        return
    data_tab, signal_tab, trade_tab, report_tab = st.tabs(
        ['数据', '指标信号', '模拟交易', '最终回测报告'])

    with data_tab:
        raw = _load_csv(source_path)
        period = raw[raw.date.between(report['start'], report['end'])]
        a, b, c = st.columns(3)
        a.metric('目标区间', '2015—2025 年')
        b.metric('交易日', f'{len(period):,}')
        c.metric('数据末日', report['end'])
        st.caption('2015—2025 年首尾都包含，共 11 个日历年；2014 年价格仅用于均线预热。')
        st.markdown(f'[查看公开数据来源]({SOURCE_URL})｜文件 SHA-256：`{digest}`')
        st.dataframe(period.assign(年份=period.date.str[:4]).groupby('年份').size().rename('交易日数').reset_index(),
                     width='stretch', hide_index=True)
        st.dataframe(period[['date', 'open', 'high', 'low', 'close', 'volume']].tail(30),
                     width='stretch', hide_index=True)
        _download(source_path, '下载原始沪深300日线')

    with signal_tab:
        st.markdown('**计算规则**：`DMA = 10日收盘均线 − 50日收盘均线`；'
                    '`AMA = DMA的10日均线`。前一日收盘若 `DMA > AMA`，次日开盘持有，否则持有现金。')
        signals_path = EXPERIMENT / 'outputs' / 'dma_signals.csv'
        signals = _load_csv(signals_path)
        chart = signals[['date', 'dma', 'ama']].set_index('date')
        st.line_chart(chart)
        st.caption('下表的 prior_close_signal 已向后移动一个交易日，表示当天开盘可执行的信号。')
        st.dataframe(signals.tail(100), width='stretch', hide_index=True)
        _download(signals_path, '下载逐日指标与信号')

    with trade_tab:
        dma_path = EXPERIMENT / 'outputs' / 'DMA_ledger.csv'
        hold_path = EXPERIMENT / 'outputs' / 'buy_and_hold_ledger.csv'
        dma, hold = _load_csv(dma_path), _load_csv(hold_path)
        dm = report['strategies']['DMA']['metrics']
        hm = report['strategies']['buy_and_hold']['metrics']
        a, b, c = st.columns(3)
        a.metric('DMA 累计净收益', f"{dm['net_return']:.2%}")
        b.metric('买入持有代理', f"{hm['net_return']:.2%}")
        c.metric('DMA 成交次数', dm['trades'])
        st.line_chart(pd.DataFrame({'DMA': dma.set_index('date').equity,
                                    '买入持有代理': hold.set_index('date').equity}))
        st.caption('初始资金 10 万元；前一日收盘信号、次日开盘成交；单边佣金 0.03%、滑点 0.02%。'
                   '指数单位允许小数，不含 ETF 分红、真实份额约束或期货保证金。')
        st.markdown('**DMA 成交记录**')
        st.dataframe(dma.loc[dma.action.ne('hold'), ['date', 'action', 'open', 'transaction_cost',
                                                       'cash', 'index_units', 'equity']],
                     width='stretch', hide_index=True)
        _download(dma_path, '下载 DMA 逐日账本')
        _download(hold_path, '下载买入持有逐日账本')

    with report_tab:
        summary_rows = []
        labels = [('DMA', 'DMA 择时'), ('buy_and_hold', '买入持有代理')]
        for key, label in labels:
            score = report['strategies'][key]['metrics']
            summary_rows.append({'策略': label, '累计净收益': f"{score['net_return']:.2%}",
                                 '年化收益': f"{score['annualized_return']:.2%}",
                                 '最大回撤': f"{score['max_drawdown']:.2%}",
                                 '年化波动': f"{score['annualized_volatility']:.2%}",
                                 '成交次数': score['trades'],
                                 '累计模拟成本(元)': round(score['total_transaction_cost'], 2)})
        st.dataframe(pd.DataFrame(summary_rows), width='stretch', hide_index=True)
        annual = pd.DataFrame({'年份': list(report['strategies']['DMA']['annual_returns']),
                               'DMA': list(report['strategies']['DMA']['annual_returns'].values()),
                               '买入持有代理': list(report['strategies']['buy_and_hold']['annual_returns'].values())})
        st.markdown('**逐年收益**')
        st.dataframe(annual.style.format({'DMA': '{:.2%}', '买入持有代理': '{:.2%}'}),
                     width='stretch', hide_index=True)
        st.markdown((EXPERIMENT / '回测报告.md').read_text(encoding='utf-8'))
        st.download_button('下载完整回测报告', (EXPERIMENT / '回测报告.md').read_bytes(),
                           'DMA_2015_2025_回测报告.md', 'text/markdown')
        st.download_button('下载参数与汇总 JSON', summary_path.read_bytes(),
                           'DMA_2015_2025_summary.json', 'application/json')
