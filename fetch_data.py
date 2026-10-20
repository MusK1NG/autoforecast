"""下载示例数据（UCI Online Retail II，45MB，两年英国批发商订单）。

    python fetch_data.py

下载到 data/online_retail_II.xlsx。第一次读取约 70 秒，之后走本地缓存。
"""
import os
import sys
import zipfile
import urllib.request

URL = 'https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip'
ROOT = os.path.dirname(os.path.abspath(__file__))
DEST_DIR = os.path.join(ROOT, 'data')
XLSX = os.path.join(DEST_DIR, 'online_retail_II.xlsx')


def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    if os.path.exists(XLSX):
        print('已存在：', XLSX)
        return
    tmp = os.path.join(DEST_DIR, '_download.zip')
    print('下载 %s …' % URL)
    urllib.request.urlretrieve(URL, tmp)
    print('解压…')
    with zipfile.ZipFile(tmp) as z:
        for name in z.namelist():
            if name.lower().endswith('.xlsx'):
                with z.open(name) as src, open(XLSX, 'wb') as dst:
                    dst.write(src.read())
    os.remove(tmp)
    print('完成：', XLSX, '%.1f MB' % (os.path.getsize(XLSX) / 1e6))


if __name__ == '__main__':
    sys.exit(main())
