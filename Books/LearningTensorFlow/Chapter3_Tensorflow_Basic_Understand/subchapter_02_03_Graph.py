import tensorflow as tf
print(tf.get_default_graph())  # tf.get_default_graph(): 어떤 그래프가 현재 기본 그래프인지 확인

g = tf.Graph()
print(g)

a = tf.constant(5)

print(a.graph is g)  # <node>.graph 속성을 통해 특정 노드가 어떤 그래프와 연결되었는지 확인
print(a.graph is tf.get_default_graph())