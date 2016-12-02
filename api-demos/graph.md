Tensorflow中的Graph
1、Tensorflow将计算表示为数据流图
2、Tensorflow中的Graph是数据操作的集合
3、Tensorflow中的Tensor是连接数据与操作(operation)的基本单元
4、Tensorflow总是默认注册一个Graph,可以通过tf.get_default_grap()方法来获取，验证方法如下：
    c = tf.constant(1.0)
    assert c.graph == tf.get_default_grap()

5、如果需要使用新的Graph，则可以通过Graph.as_default()方法来替换默认或者已经正使用的Graph，验证代码如下：
    g = tf.Graph() #创建一个新的Graph对象
    with g.as_default():#使用新的Graph对象
      c = tf.constant(1.0)
      assert c.graph == g

6、注意：tf.Graph类不是线程安全的。

7、创建并使用新的Graph如下：
    # 1. Using Graph.as_default():
    g = tf.Graph()
    with g.as_default():
    c = tf.constant(5.0)
    assert c.graph is g

    # 2. Constructing and making default:
    with tf.Graph().as_default() as g:
    c = tf.constant(5.0)
    assert c.graph is g

8、tensor(可以认为计算数据单元)和operation(操作)之间的依赖关系可以通过Graph.control_dependencies()方法来指定。
    with g.control_dependencies([a, b, c]):
    # `d` and `e` will only run after `a`, `b`, and `c` have executed.
    d = ...
    e = ...

  依赖关系还可以嵌套，如:
    with g.control_dependencies([a, b]):
      # Ops declared here run after `a` and `b`.
      with g.control_dependencies([c, d]):
        # Ops declared here run after `a`, `b`, `c`, and `d`.
  注意：必须在依赖关系的(context)上下文中构建操作，
    # WRONG
    def my_func(pred, tensor):
    t = tf.matmul(tensor, tensor)
    with tf.control_dependencies([pred]):
      # The matmul op is created outside the context, so no control
      # dependency will be added.
      return t

    # RIGHT
    def my_func(pred, tensor):
    with tf.control_dependencies([pred]):
      # The matmul op is created in the context, so a control dependency
      # will be added.
      return tf.matmul(tensor, tensor)

9、在多GPU环境中，Graph的device方法非常有用

10、Graph.name_scope()方法为上下文层的变量和操作管理提供层次化管理。参考graph-name-scope.py
    name_scope的命名必须符合如下正则表达式规范:
    [A-Za-z0-9.][A-Za-z0-9_.\\-/]* (for scopes at the root)
    [A-Za-z0-9_.\\-/]* (for other scopes)
