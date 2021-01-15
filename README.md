# event_extract_master
本人将苏神的三元组抽取算法中的DGCNN改成了事件抽取任务，并将karas改成了本人习惯使用的pytorch，在数据加载处考虑了各种语言的扩展，event extraction，django服务，train和predict

目前事件抽取支持中文和英文两种语言，中文采用百度事件抽取竞赛数据集，英文采用ACE2005数据集

在data文件夹下放置了部分数据样例，仅供参考

安装依赖：
pip3 install -r requirements.txt -i https://mirrors.ustc.edu.cn/pypi/web/simple

#启动服务

python3 manage.py runserver 0:0:0:0:8000

# 接口调用

训练：
http://localhost/apis/create_train

预测：
http://localhost/apis/event_extract



