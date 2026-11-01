from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from system.data import align_assets, load_snapshot
from system.forecast import DATA, FROZEN, OUTPUT, forecast
from system.experiments import EXPERIMENTS, list_experiments, run_experiment
from system.forward import is_pre_open_forecast
from system.gp import Config, FEATURES, make_frame
from system.model import predict_at
from system.pipeline import run, update_data
from system.robustness import run_sensitivity
from system.trend_paper_data import (OUTPUT_DIR as TREND_OUTPUT, TrendConfig, import_bars,
                                     list_datasets)
from system.trend_paper_run import list_runs as list_trend_runs, run_reproduction
from system.trend_daily import (DAILY_OUTPUT, DailyTrendConfig, list_daily_runs,
                                run_daily_transfer, training_environment)
from system.dma_ui import render_dma

PUBLIC_READ_ONLY = os.environ.get('QUANT_PUBLIC_READ_ONLY') == '1'

st.set_page_config(page_title='沪深300量化研究管理台', layout='wide')
st.title('沪深300量化研究管理台')
st.caption('数据 · 模型与策略 · 回测 · 次日预测 · 510300 ETF模拟交易｜课程研究系统')


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_signals(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def torch_environment_status() -> dict:
    return training_environment()


with st.sidebar:
    st.header('平台操作')
    if PUBLIC_READ_ONLY:
        st.info('公开演示版为只读历史研究快照；训练、上传和每日预测留在本地研究环境。')
    if st.button('更新两份日线快照', width='stretch', disabled=PUBLIC_READ_ONLY):
        try:
            with st.spinner('下载并联合校验指数与ETF行情…'):
                update_data()
            st.success('行情已更新')
        except Exception as exc:
            st.error(f'更新失败，原快照保留：{exc}')
    if st.button('生成并记录当日预测', width='stretch', disabled=PUBLIC_READ_ONLY):
        try:
            with st.spinner('计算冻结主模型与论文STGP信号…'):
                generated = forecast()
            if generated['status'] == 'ready':
                st.success('两条当日预测已生成；首次记录不可覆盖')
            else:
                st.warning('行情快照已过时限，未发布新预测')
        except Exception as exc:
            st.error(f'预测失败：{exc}')
    st.divider()
    st.caption('以下操作用于历史课程研究')
    if st.button('重算历史回测', width='stretch', disabled=PUBLIC_READ_ONLY):
        try:
            with st.spinner('重算模型与ETF账本…'):
                run()
            st.success('历史回测已更新')
        except Exception as exc:
            st.error(f'回测失败：{exc}')
    if st.button('重算参数敏感性', width='stretch', disabled=PUBLIC_READ_ONLY):
        try:
            with st.spinner('运行三组窗口与五个种子…'):
                run_sensitivity()
            st.success('敏感性检验已更新')
        except Exception as exc:
            st.error(f'敏感性检验失败：{exc}')
    st.info('本平台只管理研究预测和模拟账本，不连接券商。')

try:
    index, index_meta = load_snapshot(DATA, 'index')
    etf, etf_meta = load_snapshot(DATA, 'etf')
    index, etf = align_assets(index, etf)
except Exception as exc:
    st.error(f'行情快照校验失败：{exc}')
    st.stop()

last_date = index.index[-1]
now = datetime.now(ZoneInfo('Asia/Shanghai'))
snapshot_fresh = is_pre_open_forecast(last_date, now)
latest = read_json(OUTPUT / 'latest_forecast.json')
summary = read_json(OUTPUT / 'summary.json')
frozen = read_json(FROZEN)
signals = read_signals(OUTPUT / 'forward_signals.jsonl')
outcomes = read_csv(OUTPUT / 'forward_outcomes.csv')
current_published = bool(latest and latest.get('status') == 'ready'
                         and latest.get('data_as_of') == str(last_date.date())
                         and snapshot_fresh)
gp_runs = list_experiments()
daily_trend_runs = list_daily_runs()
if_datasets = list_datasets()
if_runs = list_trend_runs()
ar1_probability = None
if current_published:
    try:
        ar1_probability = predict_at(make_frame(index), last_date, 'AR1')
    except ValueError:
        pass

tabs = st.tabs(['总览', '数据', '模型与策略', '回测', '每日预测', '模拟交易记录',
                '遗传规划策略', '趋势混合网络复现', 'DMA择时复现'])

with tabs[0]:
    st.subheader('研究平台总览')
    a, b, c, d = st.columns(4)
    a.metric('行情截至', str(last_date.date()))
    b.metric('对齐交易日', len(index))
    c.metric('当前预测状态', '可用' if current_published else '待更新/待生成')
    d.metric('已核验前瞻记录', len(outcomes))
    if not snapshot_fresh:
        st.warning('公开演示版展示已保存的历史快照。' if PUBLIC_READ_ONLY else
                   '行情已过当前预测时限；请在下一次收盘后更新。历史记录仍可查看。')
    elif not current_published:
        st.info('行情时效合格，点击左侧“生成并记录当日预测”。')
    else:
        primary, experimental = latest['primary'], latest['experimental']
        st.success(f"主模型：{primary['direction']}（上涨概率 {primary['up_probability']:.2%}）"
                   f"｜论文STGP实验信号：{experimental['direction']}")
    st.markdown('**模型与策略状态**')
    st.dataframe(pd.DataFrame([
        {'体系': '原始AR(1)', '角色': '研究对照', '数据': '沪深300指数日线',
         '状态': '可回测；当前信号仅作非正式对照'},
        {'体系': '冻结逻辑回归', '角色': '正式主模型', '数据': '沪深300指数日线',
         '状态': '可生成前瞻预测' if snapshot_fresh else '等待新行情'},
        {'体系': '遗传规划策略', '角色': '论文方法改编／实验信号', '数据': '沪深300指数日线',
         '状态': f'已有{len(gp_runs)}次独立实验；可生成实验信号'},
        {'体系': '趋势混合网络：日线迁移', '角色': '论文方法迁移', '数据': '沪深300指数日线',
         '状态': f'已有{len(daily_trend_runs)}次训练' if daily_trend_runs else '待安装PyTorch并训练'},
        {'体系': '趋势混合网络：IF原口径', '角色': '期货方法复现', '数据': 'IF连续合约五分钟',
         '状态': f'已导入{len(if_datasets)}份，运行{len(if_runs)}次' if if_datasets else '待导入合格行情'},
        {'体系': 'DMA(10,50,10)', '角色': '2015—2025年独立择时实验', '数据': '沪深300指数日线',
         '状态': '已有逐日信号、模拟交易账本与回测报告'},
    ]), width='stretch', hide_index=True)
    st.caption('只有同为“下一交易日指数开盘至收盘方向”的日线模型可直接比较方向指标；IF五分钟目标另列。')
    if summary:
        a, b = st.columns(2)
        base = summary['results']['hw3_baseline']['retrospective']['trading']['strategy']
        gp = summary['results']['rmo']['retrospective']['trading']['strategy']
        a.metric('主模型后段ETF扣费收益', f"{base['net_return']:.2%}")
        b.metric('实验STGP后段ETF扣费收益', f"{gp['net_return']:.2%}")
        st.caption('这两个数字来自已被查看的历史区间，不代表未来收益。')

with tabs[1]:
    st.subheader('数据快照管理')
    st.markdown('**策略数据对应关系**')
    st.dataframe(pd.DataFrame([
        {'策略': 'AR(1)／冻结逻辑回归／遗传规划／趋势网络日线迁移',
         '训练与预测数据': '沪深300指数日线', '交易评价数据': '510300 ETF日线',
         '当前状态': f'两份行情对齐至{last_date.date()}'},
        {'策略': '趋势网络IF五分钟原口径', '训练与预测数据': 'IF连续合约五分钟行情',
         '交易评价数据': 'IF期货代理账本',
         '当前状态': f'已导入{len(if_datasets)}份数据' if if_datasets else '待导入2013—2018年合格数据'},
        {'策略': 'DMA(10,50,10)独立实验', '训练与预测数据': '2015—2025年沪深300指数日线',
         '交易评价数据': '指数单位代理，不是ETF', '当前状态': '结果见DMA择时复现页签'},
    ]), width='stretch', hide_index=True)
    st.caption('天天基金510300净值不作为指数开收盘价、ETF交易价或IF五分钟行情。')
    metadata = pd.DataFrame([
        {'标的': '沪深300指数', '代码': index_meta['code'], '首日': index_meta['start'],
         '末日': index_meta['end'], '行数': index_meta['rows'], '口径': index_meta['price_basis'],
         '下载时间UTC': index_meta['downloaded_at_utc']},
        {'标的': '510300 ETF', '代码': etf_meta['code'], '首日': etf_meta['start'],
         '末日': etf_meta['end'], '行数': etf_meta['rows'], '口径': etf_meta['price_basis'],
         '下载时间UTC': etf_meta['downloaded_at_utc']},
    ])
    st.dataframe(metadata, width='stretch', hide_index=True)
    st.caption('两份CSV均验证SHA-256、代码、来源、价格、OHLC、日期顺序及交易日对齐。')
    with st.expander('来源与校验值'):
        st.write(index_meta['source'])
        st.code(f"指数 SHA-256: {index_meta['sha256']}\nETF SHA-256: {etf_meta['sha256']}")
    a, b = st.columns(2)
    with a:
        st.markdown('**沪深300最近行情**')
        st.dataframe(index.tail(10).reset_index(), width='stretch', hide_index=True)
    with b:
        st.markdown('**510300 ETF最近行情**')
        st.dataframe(etf.tail(10).reset_index(), width='stretch', hide_index=True)
    dividends = read_csv(DATA / '510300_dividends.csv')
    st.markdown('**已登记ETF分红事件**')
    st.dataframe(dividends, width='stretch', hide_index=True)
    st.caption('出现新的基金分红公告时需更新分红文件，否则后续模拟收益可能漏计。')

with tabs[2]:
    st.subheader('模型与策略管理')
    st.markdown('**日线基线与正式模型**')
    st.write('AR(1)用前一交易日日内收益预测下一日收益，并从残差分布估计上涨概率，作为研究对照。'
             '正式主模型为第三次作业冻结的逻辑回归；两者使用同一目标，但AR(1)不会覆盖前瞻正式记录。')
    if frozen:
        st.markdown(f"**正式主模型**：`{frozen['winner']}`｜冻结版本 `{frozen['version']}`｜"
                    f"验证区间 {frozen['validation_start']} 至 {frozen['validation_end']}")
        candidate_rows = []
        for name, score in frozen['candidate_scores'].items():
            candidate_rows.append({'模型': name, '样本数': score['n'], '准确率': score['accuracy'],
                                   '平衡准确率': score['balanced_accuracy'], 'Brier': score['brier']})
        st.dataframe(pd.DataFrame(candidate_rows), width='stretch', hide_index=True)
        st.caption('主模型由原验证期的平衡准确率优先、Brier次之选定；本平台不会因回顾性收益自动改写冻结配置。')
    st.divider()
    a, b = st.columns(2)
    with a:
        st.markdown('### 遗传规划策略')
        st.write('对应后方“遗传规划策略”页签。强类型规则树以夏普、收益回撤比、方向正确率训练；'
                 'RMO每代加入随机目标权重，FMO是固定权重对照。默认125日训练、20日应用。')
        st.caption(f'独立历史实验 {len(gp_runs)} 次｜正式每日预测只使用已登记的RMO实验信号。'
                   '规则输出方向，不输出校准概率；当前没有论文中的独立上证指数因子。')
        if gp_runs:
            st.write(f"最近一次实验：{gp_runs[0]['id']}")
    with b:
        st.markdown('### 趋势混合网络')
        st.write('对应后方“趋势混合网络复现”页签。IF五分钟子页研究原论文期货目标；'
                 '日线迁移子页沿用平台的次日指数开盘至收盘方向目标，分别训练CNN、LSTM及融合网络。')
        st.caption(f'IF数据集 {len(if_datasets)} 份／运行 {len(if_runs)} 次；日线迁移运行 {len(daily_trend_runs)} 次。'
                   '无训练结果时不生成方向、收益或模拟账本。')
        if daily_trend_runs:
            daily_run = daily_trend_runs[0]
            st.markdown('**日线迁移：下一交易日开盘至收盘方向正确率**')
            daily_accuracy = []
            for kind, periods in daily_run['results'].items():
                for period, label in (('validation', '原验证期'), ('retrospective', '后段回顾期')):
                    direction = periods[period]['direction']
                    daily_accuracy.append({'模型': kind, '区间': label, '样本数': direction['n'],
                                           '正确率': f"{direction['accuracy']:.2%}",
                                           '预测上涨比例': f"{direction.get('signal_up_rate', 0):.2%}"})
            st.dataframe(pd.DataFrame(daily_accuracy), width='stretch', hide_index=True)
            st.caption(f"运行ID {daily_run['id']}｜数据哈希 {daily_run['source_sha256']['index']}")
            if any(row['预测上涨比例'] in ('0.00%', '100.00%') for row in daily_accuracy):
                st.warning('至少一个模型在某区间给出单一方向；其正确率不能说明模型学到了稳定的方向规律。')
        else:
            st.info('日线迁移尚未完成真实神经网络训练，暂无可报告的正确率。')
        if if_runs:
            if_run = if_runs[0]
            st.markdown('**IF五分钟原口径：下一局部趋势方向正确率**')
            if_accuracy = []
            for year, report in if_run['years'].items():
                for kind, score in report['models'].items():
                    direction = score['direction']
                    if_accuracy.append({'模型': kind, '年份': year, '样本数': direction['n'],
                                        '正确率': f"{direction['accuracy']:.2%}"})
            st.dataframe(pd.DataFrame(if_accuracy), width='stretch', hide_index=True)
            st.caption(f"运行ID {if_run['id']}｜IF五分钟结果不能与日线方向正确率直接比较。")
        else:
            st.info('IF五分钟原口径缺少合格行情和训练结果，暂无可报告的正确率。')
    st.divider()
    if current_published:
        e = latest['experimental']
        st.info(f"当前规则 {e['rule_id']}｜窗口起点 {e['block_start']}｜训练 {e['train_start']} 至 {e['train_end']}｜种子 {e['seed']}")
    windows = read_csv(OUTPUT / 'rmo_windows.csv')
    if not windows.empty:
        st.markdown('**历史滚动窗口及规则**')
        st.dataframe(windows.drop(columns=['rule'], errors='ignore'), width='stretch', hide_index=True)
        with st.expander('查看原始规则表达式'):
            st.dataframe(windows[['block', 'rule_id', 'rule']], width='stretch', hide_index=True)

with tabs[3]:
    st.subheader('同目标日线策略回测')
    if summary is None:
        st.info('尚无历史结果，请点击左侧“重算历史回测”。')
    else:
        rows = []
        names = [('ar1', 'AR(1)对照'), ('hw3_baseline', '冻结逻辑回归'),
                 ('rmo', '遗传规划RMO'), ('fmo', '遗传规划FMO')]
        names = [(key, name) for key, name in names if key in summary['results']]
        periods = [('validation', '原验证期'), ('retrospective', '后段回顾期')]
        for key, name in names:
            for period, label in periods:
                score = summary['results'][key][period]
                forecast_score = score['forecast']
                trading = score['trading']['strategy']
                rows.append({'模型': name, '区间': label, '预测数': forecast_score['n'],
                             '方向准确率': forecast_score['accuracy'],
                             '平衡准确率': forecast_score['balanced_accuracy'],
                             'ETF扣费收益': trading['net_return'],
                             '最大回撤': trading['max_drawdown'],
                             '交易次数': trading['trades'], '交易成本(元)': trading['total_cost']})
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        st.caption('以上模型均预测次日沪深300指数开盘至收盘方向；ETF收益来自同日但不同持仓区间的T+1模拟。'
                   '后段历史已被查看，不是新的盲测。')
        if daily_trend_runs:
            st.markdown('**趋势混合网络日线迁移实验**')
            selected_daily = st.selectbox('选择日线迁移运行',
                                          [record['id'] for record in daily_trend_runs],
                                          key='backtest_daily_run')
            daily_report = next(record for record in daily_trend_runs if record['id'] == selected_daily)
            daily_rows = []
            for kind, scores in daily_report['results'].items():
                for period, label in periods:
                    score = scores[period]
                    daily_rows.append({'模型': kind, '区间': label,
                                       '预测数': score['direction']['n'],
                                       '方向准确率': score['direction']['accuracy'],
                                       '平衡准确率': score['direction']['balanced_accuracy'],
                                       'ETF扣费收益': score['trading']['strategy']['net_return'],
                                       '最大回撤': score['trading']['strategy']['max_drawdown']})
            st.dataframe(pd.DataFrame(daily_rows), width='stretch', hide_index=True)
            st.caption('日线方法迁移使用同一方向目标；配置、阈值和数据哈希见趋势混合网络页签。')
        else:
            st.info('趋势混合网络日线迁移尚无真实训练结果，不显示占位成绩。')
        chosen_period = st.selectbox('资金曲线区间', ['retrospective', 'validation'],
                                     format_func=lambda x: '后段回顾期' if x == 'retrospective' else '原验证期')
        curves = {}
        for key, name in names:
            ledger = read_csv(OUTPUT / f'{key}_{chosen_period}_etf_ledger.csv')
            if not ledger.empty:
                curves[name] = ledger.set_index('date')['equity']
        if curves:
            st.line_chart(pd.DataFrame(curves))
        sensitivity = read_csv(OUTPUT / 'robustness_summary.csv')
        if not sensitivity.empty:
            st.markdown('**窗口与随机种子敏感性**')
            st.dataframe(sensitivity, width='stretch', hide_index=True)
            st.caption('三组窗口各五个种子，使用同一历史数据，属于诊断性检验。')
    st.markdown('**IF五分钟期货原口径**')
    if if_runs:
        st.info(f'已有 {len(if_runs)} 次独立期货方法运行；目标与日线方向不同，详细逐年结果见“趋势混合网络复现”。')
    else:
        st.info('尚无合格IF五分钟数据与训练结果，不展示期货收益。')

with tabs[4]:
    st.subheader('每日预测与结果核验')
    st.caption('正式发布：冻结逻辑回归；并行实验信号：遗传规划RMO。AR(1)仅显示当前研究对照，不写入首次预测日志。'
               '趋势网络日线迁移尚未冻结为每日模型；IF五分钟期货目标不进入本页的指数预测。')
    if current_published:
        primary, experimental = latest['primary'], latest['experimental']
        a, b, c = st.columns(3)
        a.metric('主模型方向', primary['direction'])
        b.metric('主模型上涨概率', f"{primary['up_probability']:.2%}")
        c.metric('论文STGP实验方向', experimental['direction'])
        st.caption(f"信号日 {latest['data_as_of']}｜主模型版本 {latest['primary_version']}｜"
                   f"STGP规则 {experimental['rule_id']}｜首次预测记录不可覆盖")
        if primary['signal_up'] != experimental['signal_up']:
            st.info('两模型方向不同；当前正式预测以冻结主模型为准。')
        if ar1_probability is not None:
            st.markdown('**AR(1)非正式对照**')
            st.write(f"同一快照计算的上涨概率 {ar1_probability:.2%}，方向"
                     f"{'上涨' if ar1_probability >= 0.5 else '下跌或持平'}；"
                     '该值未发布为正式预测，也不写入不可覆盖的前瞻记录。')
    else:
        st.warning('目前没有时效合格的当前预测。请更新行情并生成新信号。')
    if daily_trend_runs:
        st.caption(f"趋势网络最近日线实验 {daily_trend_runs[0]['id']} 仅用于历史回顾；未接入每日预测。")
    else:
        st.caption('趋势混合网络日线迁移待训练；当前不生成实时信号。')
    st.markdown('**前瞻预测留痕**')
    if signals.empty:
        st.info('尚无首次预测记录。')
    else:
        st.dataframe(signals.sort_values('recorded_at', ascending=False).head(50), width='stretch', hide_index=True)
        st.download_button('下载首次预测日志', (OUTPUT / 'forward_signals.jsonl').read_bytes(),
                           'forward_signals.jsonl', 'application/json')
    st.markdown('**已核验结果**')
    if outcomes.empty:
        st.info('下一交易日结果尚未写入行情快照。')
    else:
        st.dataframe(outcomes.sort_values('target_date', ascending=False).head(50), width='stretch', hide_index=True)
        st.download_button('下载结果核验', (OUTPUT / 'forward_outcomes.csv').read_bytes(),
                           'forward_outcomes.csv', 'text/csv')

with tabs[5]:
    st.subheader('510300 ETF模拟交易记录')
    st.caption('仅展示同目标日线策略的历史ETF账本：次日开盘模拟成交、100份取整、T+1卖出，'
               '计入佣金、滑点和已登记分红。IF五分钟期货账本与此分开。')
    model_options = {
        'AR(1)研究对照': ('fixed', 'ar1'),
        '冻结逻辑回归': ('fixed', 'hw3_baseline'),
        '遗传规划RMO固定回测': ('fixed', 'rmo'),
        '遗传规划FMO固定回测': ('fixed', 'fmo'),
    }
    for item in gp_runs:
        for method in item['methods']:
            model_options[f"遗传规划{method}实验｜{item['id']}"] = ('gp', item['id'], method.lower())
    for item in daily_trend_runs:
        for kind in item['results']:
            model_options[f"趋势网络日线{kind}｜{item['id']}"] = ('daily', item['id'], kind)
    model_title = st.selectbox('策略', list(model_options))
    period_title = st.selectbox('账本区间', ['后段回顾期', '原验证期'])
    period = 'retrospective' if period_title == '后段回顾期' else 'validation'
    choice = model_options[model_title]
    if choice[0] == 'fixed':
        ledger_path = OUTPUT / f'{choice[1]}_{period}_etf_ledger.csv'
    elif choice[0] == 'gp':
        ledger_path = EXPERIMENTS / choice[1] / f'{choice[2]}_{period}_etf_ledger.csv'
    else:
        ledger_path = DAILY_OUTPUT / choice[1] / f'{choice[2]}_{period}_etf_ledger.csv'
    ledger = read_csv(ledger_path)
    if ledger.empty:
        st.info('尚无此策略的模拟账本，请先重算历史回测。')
    else:
        buys = int(ledger.action.eq('buy').sum())
        sells = int(ledger.action.eq('sell').sum())
        costs = float((ledger.commission + ledger.slippage).sum())
        a, b, c = st.columns(3)
        a.metric('模拟买入次数', buys)
        b.metric('模拟卖出次数', sells)
        c.metric('累计交易成本', f'{costs:,.2f} 元')
        actions = st.multiselect('记录类型', ['buy', 'sell', 'hold'], default=['buy', 'sell', 'hold'])
        filtered = ledger[ledger.action.isin(actions)]
        st.dataframe(filtered.sort_values('date', ascending=False), width='stretch', hide_index=True)
        st.line_chart(ledger.set_index('date')['equity'])
        st.download_button('下载完整模拟账本', ledger_path.read_bytes(), ledger_path.name, 'text/csv')
        st.info('账本是课程模拟，不代表实际可按开盘价全部成交，也不会产生真实订单。')

with tabs[6]:
    st.subheader('遗传规划策略训练与验证')
    st.info('强类型规则树按训练窗口进化：用收益、夏普和最大回撤等目标评价规则；RMO加入随机权重，FMO使用固定权重。每段规则在之后的应用窗口生成次日开盘至收盘方向，并与冻结主模型在相同日期比较。')
    st.caption('默认训练125个已完成标签的交易日，向后应用20个信号日；可勾选输入特征并调整种群、代数、树深和随机种子。')
    st.warning('此处的参数试验只在已查看的历史上回顾验证，不会改变正式每日预测、冻结主模型或作业固定回测。')
    with st.expander('查看11个可选特征的含义'):
        st.dataframe(pd.DataFrame([
            ('ar_1', '当日指数收盘对前收盘的对数收益'),
            ('ar_2', '前一交易日指数对数收益'),
            ('intraday_1', '当日开盘至收盘对数收益'),
            ('return_5', '近5日累计对数收益'),
            ('return_20', '近20日累计对数收益'),
            ('vol_5', '近5日收益波动率'),
            ('vol_20', '近20日收益波动率'),
            ('range_1', '当日高低价区间除以收盘价'),
            ('range_5', '近5日平均价格区间'),
            ('volume_ratio', '当日成交量相对近20日均量，中心化后输入STGP'),
            ('ma_gap', '5日均线相对20日均线的偏离'),
        ], columns=['特征', '含义']), width='stretch', hide_index=True)
    with st.form('experiment_config'):
        features = st.multiselect('规则可使用的特征', list(FEATURES), default=list(FEATURES),
                                  help='仅能选择信号日收盘时已知的内置特征；规则生成与变异都受此集合约束。')
        methods = st.multiselect('适应度方法', ['RMO', 'FMO'], default=['RMO', 'FMO'],
                                 help='RMO每代加入随机多目标权重；FMO使用固定权重。')
        a, b, c = st.columns(3)
        with a:
            train_days = st.number_input('训练窗口（日）', 60, 250, 125, step=5)
            predict_days = st.number_input('规则应用窗口（日）', 5, 40, 20, step=5)
        with b:
            population = st.number_input('种群规模', 8, 200, 100, step=4)
            generations = st.number_input('进化代数', 2, 40, 20, step=1)
        with c:
            max_depth = st.number_input('最大树深', 3, 7, 5, step=1)
            seed = st.number_input('随机种子', 0, 2_000_000_000, 20261026, step=1)
        submitted = st.form_submit_button('训练规则并验证策略', width='stretch', disabled=PUBLIC_READ_ONLY)
    if submitted:
        try:
            config = Config(train_days=int(train_days), predict_days=int(predict_days),
                            population=int(population), generations=int(generations),
                            max_depth=int(max_depth), seed=int(seed), features=tuple(features))
            with st.spinner('正在按所选特征与方法滚动训练、回测并保存…'):
                result = run_experiment(config, tuple(methods))
            st.success(f"实验已保存：{result['id']}")
        except Exception as exc:
            st.error(f'实验未完成，未发布结果：{exc}')
    history = list_experiments()
    st.markdown('**历史实验记录**')
    if not history:
        st.info('尚无动态实验。上方默认配置对应论文启发的125／20方法。')
    else:
        selected_id = st.selectbox('选择实验', [item['id'] for item in history])
        selected = next(item for item in history if item['id'] == selected_id)
        st.caption(f"运行时间UTC {selected['created_at_utc']}｜数据哈希：{selected['source_sha256']}")
        st.json({'config': selected['config'], 'methods': selected['methods']}, expanded=False)
        comparison = []
        for method, values in selected['results'].items():
            for period in ('validation', 'retrospective'):
                score = values[period]
                comparison.append({'方法': method, '区间': '原验证期' if period == 'validation' else '后段回顾期',
                                   '样本数': score['forecast']['n'],
                                   '方向准确率': score['forecast']['accuracy'],
                                   '平衡准确率': score['forecast']['balanced_accuracy'],
                                   'ETF扣费收益': score['trading']['strategy']['net_return'],
                                   '最大回撤': score['trading']['strategy']['max_drawdown'],
                                   '交易次数': score['trading']['strategy']['trades']})
        st.dataframe(pd.DataFrame(comparison), width='stretch', hide_index=True)
        if summary:
            baseline = []
            for period in ('validation', 'retrospective'):
                score = summary['results']['hw3_baseline'][period]
                baseline.append({'区间': '原验证期' if period == 'validation' else '后段回顾期',
                                 '主模型平衡准确率': score['forecast']['balanced_accuracy'],
                                 '主模型ETF扣费收益': score['trading']['strategy']['net_return']})
            st.markdown('**冻结主模型固定对照**')
            st.dataframe(pd.DataFrame(baseline), width='stretch', hide_index=True)
        experiment_path = EXPERIMENTS / selected_id
        for method in selected['methods']:
            windows_path = experiment_path / f'{method.lower()}_windows.csv'
            with st.expander(f'{method}滚动窗口与规则'):
                st.dataframe(read_csv(windows_path), width='stretch', hide_index=True)
            st.download_button(f'下载{method}逐日预测',
                               (experiment_path / f'{method.lower()}_predictions.csv').read_bytes(),
                               f'{selected_id}_{method.lower()}_predictions.csv', 'text/csv')
        st.download_button('下载实验配置与结果JSON',
                           (experiment_path / 'summary.json').read_bytes(),
                           f'{selected_id}_summary.json', 'application/json')

with tabs[7]:
    paper_tab, daily_tab = st.tabs(['IF五分钟原口径', '日线趋势迁移'])
    torch_status = torch_environment_status()
    if torch_status['ready']:
        st.success(f"神经网络训练环境已就绪：PyTorch {torch_status['torch_version']}｜{torch_status['python']}")
    else:
        st.warning(f"神经网络训练环境尚未就绪：{torch_status['reason']}")
    with paper_tab:
        st.subheader('趋势混合网络方法复现')
        st.info('论文使用沪深300股指期货5分钟连续价格：自上而下分割局部趋势，最近4段趋势送LSTM、对应4×5价格送CNN，融合后分别预测下一段方向与持续时间。与单独CNN、LSTM对照，按2016—2018年逐年回顾验证。')
        st.warning('本模块与指数日线的次日方向预测、510300 ETF账本相互独立。当前没有论文所需的IF五分钟历史数据；导入前不会显示复现收益。')
        with st.expander('输入格式、口径与实现假设'):
            st.code('timestamp,close,contract\n2013-01-04 09:20:00,2500.4,IF1301\n2013-01-04 09:25:00,2501.0,IF1301')
            st.write('接受CSV或Parquet；时间戳为上海本地时间、五分钟整点、严格升序。须提供已连续化并说明调整方式的IF数据，覆盖2013—2018年。'
                     '论文未完整给出连续合约构造和分段阈值，因此这些参数在每次运行中显式保存。')
            st.write('模拟使用一倍名义本金多空、五分钟收盘价成交、收盘价观察0.6%止损及论文费率代理；'
                     '缺少盘口与逐笔数据，不能据此声称可实盘成交。')
        with st.form('trend_import'):
            uploaded = st.file_uploader('导入IF五分钟连续合约数据', type=['csv', 'parquet'],
                                        disabled=PUBLIC_READ_ONLY)
            source_note = st.text_input('数据来源', placeholder='供应商、下载日期及许可说明')
            adjustment_note = st.text_input('连续合约与价格调整说明', placeholder='主力切换规则、前复权或后复权方式')
            import_clicked = st.form_submit_button('校验并保存数据集', disabled=PUBLIC_READ_ONLY)
        if import_clicked:
            try:
                if uploaded is None:
                    raise ValueError('请先选择CSV或Parquet文件')
                imported = import_bars(uploaded.getvalue(), uploaded.name, source_note, adjustment_note)
                st.success(f"数据集已保存：{imported['id']}；完整论文覆盖：{'是' if imported['coverage']['complete_for_paper'] else '否'}")
            except Exception as exc:
                st.error(f'导入失败，未发布数据集：{exc}')
        datasets = list_datasets()
        st.markdown('**已导入数据集**')
        if datasets:
            st.dataframe(pd.DataFrame([{'ID': d['id'], '来源': d['source'], '行数': d['coverage']['rows'],
                                        '首日': d['coverage']['first'], '末日': d['coverage']['last'],
                                        '论文覆盖完整': d['coverage']['complete_for_paper'],
                                        '合约切换次数': d['coverage']['contract_switches']}
                                       for d in datasets]), width='stretch', hide_index=True)
            dataset_id = st.selectbox('本次复现使用的数据集', [d['id'] for d in datasets])
            selected_dataset = next(d for d in datasets if d['id'] == dataset_id)
            st.caption(f"来源哈希：{selected_dataset['source_sha256']}｜连续化：{selected_dataset['adjustment']}")
        else:
            dataset_id = None
            selected_dataset = None
            st.info('待导入：尚无IF五分钟数据，不会生成论文复现指标或交易收益。')
        with st.form('trend_run'):
            a, b, c = st.columns(3)
            with a:
                error_pct = st.number_input('趋势分段相对误差', 0.0001, 0.03, 0.002, step=0.0001, format='%.4f')
                min_bars = st.number_input('最短趋势K线数', 3, 30, 4)
                eval_step = st.number_input('测试预测间隔（根K线）', 1, 72, 12)
            with b:
                epochs = st.number_input('训练轮次', 1, 100, 15)
                batch = st.number_input('批量', 16, 1024, 128, step=16)
                learning_rate = st.number_input('学习率', 0.00001, 0.05, 0.005, step=0.0001, format='%.5f')
            with c:
                hidden_size = st.number_input('LSTM隐藏单元', 16, 256, 100, step=4)
                max_hold = st.number_input('最大持仓K线数', 1, 144, 72)
                trend_seed = st.number_input('复现随机种子', 0, 2_000_000_000, 20261031)
            shuffle_training = st.checkbox('训练时打乱样本顺序', value=False,
                                           help='论文策略实验采用不打乱；勾选可做稳定性对照。')
            submitted_trend = st.form_submit_button('运行三年论文方法复现', disabled=PUBLIC_READ_ONLY or dataset_id is None or not selected_dataset['coverage']['complete_for_paper'])
        if submitted_trend:
            try:
                cfg = TrendConfig(error_pct=float(error_pct), min_segment_bars=int(min_bars),
                                  epochs=int(epochs), batch_size=int(batch), learning_rate=float(learning_rate),
                                  hidden_size=int(hidden_size), evaluation_step_bars=int(eval_step),
                                  max_hold_bars=int(max_hold), seed=int(trend_seed),
                                  shuffle_training=bool(shuffle_training))
                with st.spinner('逐年训练CNN、LSTM与混合模型并保存预测和模拟交易…'):
                    trend_result = run_reproduction(dataset_id, cfg)
                st.success(f"复现运行完成：{trend_result['id']}")
            except Exception as exc:
                st.error(f'复现未完成：{exc}')
        trend_runs = list_trend_runs()
        st.markdown('**论文复现历史运行**')
        if not trend_runs:
            st.info('尚无真实IF五分钟数据训练结果。论文原表格数字不会作为本系统成绩填入。')
        else:
            trend_id = st.selectbox('查看运行', [r['id'] for r in trend_runs])
            record = next(r for r in trend_runs if r['id'] == trend_id)
            st.caption(f"数据集 {record['dataset_id']}｜来源SHA-256 {record['source_sha256']}")
            st.write(record['caveat'])
            score_rows = []
            for year, report in record['years'].items():
                for name, values in report['models'].items():
                    score_rows.append({'年份': year, '模型': name,
                                       '方向准确率': values['direction']['accuracy'],
                                       '精确率': values['direction']['precision'],
                                       '召回率': values['direction']['recall'],
                                       'F0.4': values['direction']['f_0_4'],
                                       '持续时间RMSE（小时）': values['duration_rmse_hours'],
                                       '模拟扣费收益': values['trading']['net_return'],
                                       '最大回撤': values['trading']['max_drawdown'],
                                       '买入持有代理收益': report['buy_and_hold_proxy']['net_return']})
            st.dataframe(pd.DataFrame(score_rows), width='stretch', hide_index=True)
            run_path = TREND_OUTPUT / trend_id
            st.download_button('下载逐年预测', (run_path / 'predictions.csv').read_bytes(),
                               f'{trend_id}_predictions.csv', 'text/csv')
            st.download_button('下载运行配置及指标', (run_path / 'summary.json').read_bytes(),
                               f'{trend_id}_summary.json', 'application/json')

    with daily_tab:
        st.subheader('日线趋势方法迁移实验')
        st.info('使用平台已校验的沪深300指数日线训练CNN＋LSTM趋势结构，预测下一交易日开盘至收盘方向；另预测下一局部趋势的持续交易日。510300 ETF只用于同日期模拟账本。')
        st.warning('这是日线方法迁移，不是IF五分钟期货论文原口径复现。天天基金510300公开净值是基金每日净值，不等于交易所ETF开收盘价或IF期货五分钟价格，因此当前不将其混入训练特征。')
        st.caption(f"指数行情截至 {index_meta['end']}｜来源SHA-256 {index_meta['sha256']}｜训练标签截止 2025-12-31")
        with st.form('daily_trend_config'):
            a, b, c = st.columns(3)
            with a:
                daily_error = st.number_input('日线分段相对误差', 0.0001, 0.03, 0.01, step=0.0005, format='%.4f')
                daily_min_days = st.number_input('最短趋势交易日', 3, 30, 4)
            with b:
                daily_epochs = st.number_input('日线训练轮次', 1, 100, 15)
                daily_batch = st.number_input('日线训练批量', 16, 1024, 128, step=16)
                daily_learning_rate = st.number_input('日线学习率', 0.00001, 0.05, 0.005, step=0.0001, format='%.5f')
            with c:
                daily_hidden = st.number_input('日线LSTM隐藏单元', 16, 256, 100, step=4)
                daily_seed = st.number_input('日线随机种子', 0, 2_000_000_000, 20261031)
                daily_shuffle = st.checkbox('日线训练打乱样本', value=False)
            daily_submitted = st.form_submit_button('运行并保存日线迁移实验', disabled=PUBLIC_READ_ONLY)
        if daily_submitted:
            try:
                cfg = DailyTrendConfig(error_pct=float(daily_error), min_segment_days=int(daily_min_days),
                                       epochs=int(daily_epochs), batch_size=int(daily_batch),
                                       learning_rate=float(daily_learning_rate), hidden_size=int(daily_hidden),
                                       seed=int(daily_seed), shuffle_training=bool(daily_shuffle))
                with st.spinner('按2025年底前数据训练三种网络，评价2026年两段历史…'):
                    daily_result = run_daily_transfer(cfg)
                st.success(f"日线迁移实验已保存：{daily_result['id']}")
            except Exception as exc:
                st.error(f'日线迁移实验未完成：{exc}')
        daily_runs = list_daily_runs()
        st.markdown('**历史日线迁移实验**')
        if not daily_runs:
            st.info('尚无日线迁移结果；运行需要安装额外的PyTorch依赖。')
        else:
            daily_id = st.selectbox('查看日线实验', [r['id'] for r in daily_runs])
            report = next(r for r in daily_runs if r['id'] == daily_id)
            st.caption(f"训练样本 {report['training']['samples']}｜指数SHA-256 {report['source_sha256']['index']}")
            st.write(report['warning'])
            rows = []
            for kind, periods in report['results'].items():
                for period in ('validation', 'retrospective'):
                    score = periods[period]
                    rows.append({'模型': kind, '区间': '原验证期' if period == 'validation' else '后段回顾期',
                                 '预测数': score['direction']['n'],
                                 '方向准确率': score['direction']['accuracy'],
                                 '平衡准确率': score['direction']['balanced_accuracy'],
                                 'Brier': score['direction']['brier'],
                                 '时长RMSE（交易日）': score['duration_rmse_days'],
                                 'ETF扣费收益': score['trading']['strategy']['net_return'],
                                 '最大回撤': score['trading']['strategy']['max_drawdown']})
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
            if summary:
                baseline = []
                for period in ('validation', 'retrospective'):
                    score = summary['results']['hw3_baseline'][period]
                    baseline.append({'区间': '原验证期' if period == 'validation' else '后段回顾期',
                                     '主模型平衡准确率': score['forecast']['balanced_accuracy'],
                                     '主模型ETF扣费收益': score['trading']['strategy']['net_return']})
                st.markdown('**冻结主模型固定对照**')
                st.dataframe(pd.DataFrame(baseline), width='stretch', hide_index=True)
            selected_path = DAILY_OUTPUT / daily_id
            st.download_button('下载日线实验配置与结果', (selected_path / 'summary.json').read_bytes(),
                               f'{daily_id}_summary.json', 'application/json')
            for kind in report['results']:
                st.download_button(f'下载{kind}逐日预测', (selected_path / f'{kind}_predictions.csv').read_bytes(),
                                   f'{daily_id}_{kind}_predictions.csv', 'text/csv')

with tabs[8]:
    render_dma()
