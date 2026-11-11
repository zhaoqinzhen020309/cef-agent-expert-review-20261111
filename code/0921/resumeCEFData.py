"""显式补发被输出长度截断的Data请求，再继续原实验。"""
import sys
sys.dont_write_bytecode=True
from scripts.cef_data_resume import main

if __name__=='__main__':raise SystemExit(main())
