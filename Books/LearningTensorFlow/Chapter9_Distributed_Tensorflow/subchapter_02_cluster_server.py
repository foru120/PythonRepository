import tensorflow as tf

parameter_servers = ['localhost:2222']
workers = ['localhost:2223',
           'localhost:2224',
           'localhost:2225']

# 텐서플로우 클러스터 생성
cluster = tf.train.ClusterSpec({'parameter_server': parameter_servers,

                                'worker': workers})
# 첫 번째 워커 노드 서버 생성
server = tf.train.Server(cluster,
                         job_name='worker',
                         task_index=0)

