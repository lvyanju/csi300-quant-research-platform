"""Streamlit Community Cloud 入口：访客只能查看已保存的研究记录。"""
import os

os.environ['QUANT_PUBLIC_READ_ONLY'] = '1'

import app  # noqa: E402,F401  Streamlit 页面由导入执行
