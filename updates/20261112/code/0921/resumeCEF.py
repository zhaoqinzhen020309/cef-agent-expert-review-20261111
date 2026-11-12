"""显式重试已记录的网络失败请求，再继续原实验。"""
import sys
sys.dont_write_bytecode=True
from scripts.cef_network_resume import main

if __name__=='__main__':
    raise SystemExit(main())
