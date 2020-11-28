# EventTrainServer
百度竞赛事件抽取，本人将苏神的三元组抽取算法中的DGCNN改成了事件抽取任务，并将karas改成了本人习惯使用的pytorch，在数据加载处考虑了各种语言的扩展，event extraction，django服务，train和predict

项目依赖pytorch==1.3.0
mlflow
jieba
nltk
、、、、

#启动服务

python3 manage.py runserver 0:0:0:0:8000

# 接口调用

训练：
http://localhost/apis/create_train

预测：
http://localhost/apis/event_extract



