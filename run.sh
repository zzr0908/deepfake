curl -o multiFFDI-phase1.tar.gz 'http://zoloz-open.oss-cn-hangzhou.aliyuncs.com/waitan2024_deepfake_challenge%2F_%E8%B5%9B%E9%81%931%E5%AF%B9%E5%A4%96%E5%8F%91%E5%B8%83%E6%95%B0%E6%8D%AE%E9%9B%86%2Fphase1.tar.gz?Expires=1726603663&OSSAccessKeyId=LTAI5tAfcZDV5eCa1BBEJL9R&Signature=wFrzBHn5bhULqWzlZP7Z74p1g9c%3D'
tar -xzvf multiFFDI-phase1.tar.gz

# Step 2: 安装依赖包
pip install -r requirements.txt

# Step 3: 文件预处理
python preprocess.py