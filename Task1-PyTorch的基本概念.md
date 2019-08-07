
# 1.PyTorch概念

(1)PyTorch是使用GPU和CPU优化的深度学习张量库。

(2)PyTorch是一个基于python的科学计算包，主要定位两类人群：
* NumPy的替代品，可以利用GPU的性能进行计算
* 深度学习研究平台拥有足够的灵活性和速度

PyTorch不仅定义网络结构简单，而且还很直观灵活。静态图的网络定义都是声明式的，而动态图可以随意的调用函数（if，for，list什么的随便用），两者的差距不是一点点。网络的定义在任何框架中都应该是属于最基础最简单的一个内容，即使是接口繁多的tensorflow，通过不断的查文档，新手也能把模型搭起来或者是看懂别人的模型。这个PyTorch也不例外，它的优势在于模型定义十分直观易懂，很容易看懂看别人写的代码。

优势：

* 它属于轻量级；

* 它使你能够明确地控制计算。没有编译器能自己妄图变聪明来「帮助你」，或是将你的代码加速；事实上大多编译器在调试中会产生大量麻烦；

* 它使 GPU 内核调用之上仅有少量（可解释的）抽象层，而这恰恰是高性能的保证；

* 更容易调试，因为你可以只使用标准的 PyThon 工具；

* PyTorch 让自定义的实现更加容易，所以你得以将更多时间专注于算法中，这样往往能够改进主要性能；

* Torch-vision 使加载和变换图像变得容易。

* PyTorch 提供了一个强化功能。增强功能基本上不会在实现中产生过多资源消耗，能有一些内置函数来调用 RL 的感觉真棒。



# 2.Pytroch的安装

安装python及配置环境比较基础，这里直接使用anaconda的套装，安装好anaconda之后，直接在prompt里输入命令`conda install pytorch torchvision cudatoolkit=9.0 -c pytorch`安装对应版本的pytorch。这里安装的版本的基于windows的python3.6版本。

# 3.通用代码实现流程

## 3.1Tensors (张量)
Tensors 类似于 NumPy 的 ndarrays ，同时  Tensors 可以使用 GPU 进行计算。首先构造一个5x3矩阵，不初始化。


```python
import torch

x = torch.empty(5, 3)
print(x)
```

    tensor([[1.9264e+21, 4.5880e-41, 1.9264e+21],
            [4.5880e-41, 1.8465e+25, 1.9857e+29],
            [7.5556e+31, 3.0881e+29, 1.4607e-19],
            [2.0333e+32, 7.0969e+22, 1.7409e+25],
            [1.4602e-19, 1.1257e+24, 1.8672e+25]])


构造一个随机初始化的矩阵：


```python
x = torch.rand(5, 3)
print(x)
```

    tensor([[0.2731, 0.9215, 0.8159],
            [0.3026, 0.9387, 0.7976],
            [0.6585, 0.0352, 0.9535],
            [0.8510, 0.5159, 0.4770],
            [0.6505, 0.5207, 0.9406]])


构造一个矩阵全为 0，而且数据类型是 long.


```python
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
```

    tensor([[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]])


构造一个张量，直接使用数据：


```python
x = torch.tensor([5.5, 3])
print(x)
```

    tensor([5.5000, 3.0000])


创建一个 tensor 基于已经存在的 tensor。


```python
#创建一个全为1的5*3矩阵
x = x.new_ones(5, 3, dtype=torch.float)      
print(x)

#创建一个和x有相同size的随机初始化的矩阵
y = torch.randn_like(x, dtype=torch.float)
print(y)

#获取它的维度信息
print(x.size())
print(y.size())


#加法，方式一
print(x + y)
#加法，方式二
print(torch.add(x, y))
```

    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]])
    tensor([[-0.5432,  1.1691, -1.0323],
            [-0.4143,  0.2582, -1.4747],
            [-1.1857,  0.3030, -2.2335],
            [-2.2696,  0.6668,  0.3202],
            [ 0.8481,  1.2461,  0.9751]])
    torch.Size([5, 3])
    torch.Size([5, 3])
    tensor([[ 0.4568,  2.1691, -0.0323],
            [ 0.5857,  1.2582, -0.4747],
            [-0.1857,  1.3030, -1.2335],
            [-1.2696,  1.6668,  1.3202],
            [ 1.8481,  2.2461,  1.9751]])
    tensor([[ 0.4568,  2.1691, -0.0323],
            [ 0.5857,  1.2582, -0.4747],
            [-0.1857,  1.3030, -1.2335],
            [-1.2696,  1.6668,  1.3202],
            [ 1.8481,  2.2461,  1.9751]])


加法: 提供一个输出 tensor 作为参数


```python
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
```

    tensor([[ 0.4568,  2.1691, -0.0323],
            [ 0.5857,  1.2582, -0.4747],
            [-0.1857,  1.3030, -1.2335],
            [-1.2696,  1.6668,  1.3202],
            [ 1.8481,  2.2461,  1.9751]])


加法: in-place


```python
# adds x to y
y.add_(x)
print(y)
```

    tensor([[ 0.4568,  2.1691, -0.0323],
            [ 0.5857,  1.2582, -0.4747],
            [-0.1857,  1.3030, -1.2335],
            [-1.2696,  1.6668,  1.3202],
            [ 1.8481,  2.2461,  1.9751]])


注意:任何使张量会发生变化的操作都有一个前缀。例如：`x.copy(y)`, `x.t_()`, 将会改变 x.

你可以使用标准的  NumPy 类似的索引操作


```python
print(y[:, 1])
```

    tensor([2.1691, 1.2582, 1.3030, 1.6668, 2.2461])


改变大小：如果你想改变一个 tensor 的大小或者形状，你可以使用 `torch.view`:


```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())
```

    torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])


如果你有一个元素 tensor ，使用 .item() 来获得这个 value 。


```python
x = torch.randn(1,3)
print(x)
print(x[0,1].item())
```

    tensor([[-0.8989, -1.1348,  1.5393]])
    -1.1348298788070679


## 3.2自动微分

autograd 包是 PyTorch 中所有神经网络的核心。首先让我们简要地介绍它，然后我们将会去训练我们的第一个神经网络。该 autograd 软件包为 Tensors 上的所有操作提供自动微分。它是一个由运行定义的框架，这意味着以代码运行方式定义你的后向传播，并且每次迭代都可以不同。我们从 tensor 和 gradients 来举一些例子。

1、TENSOR

torch.Tensor 是包的核心类。如果将其属性 .requires_grad 设置为 True，则会开始跟踪针对 tensor 的所有操作。完成计算后，您可以调用 .backward() 来自动计算所有梯度。该张量的梯度将累积到 .grad 属性中。

要停止 tensor 历史记录的跟踪，您可以调用 .detach()，它将其与计算历史记录分离，并防止将来的计算被跟踪。

要停止跟踪历史记录（和使用内存），您还可以将代码块使用 with torch.no_grad(): 包装起来。在评估模型时，这是特别有用，因为模型在训练阶段具有 requires_grad = True 的可训练参数有利于调参，但在评估阶段我们不需要梯度。

还有一个类对于 autograd 实现非常重要那就是 Function。Tensor 和 Function 互相连接并构建一个非循环图，它保存整个完整的计算过程的历史信息。每个张量都有一个 .grad_fn 属性保存着创建了张量的 Function 的引用，（如果用户自己创建张量，则g rad_fn 是 None ）。

如果你想计算导数，你可以调用 Tensor.backward()。如果 Tensor 是标量（即它包含一个元素数据），则不需要指定任何参数backward()，但是如果它有更多元素，则需要指定一个gradient 参数来指定张量的形状。


```python
import torch

#创建一个张量，设置 requires_grad=True 来跟踪与它相关的计算
x = torch.ones(2, 2, requires_grad=True)
print(x)
```

    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)



```python
y = x + 2
print(y)
print(y.grad_fn)
```

    tensor([[3., 3.],
            [3., 3.]], grad_fn=<AddBackward0>)
    <AddBackward0 object at 0x7fe543ce90b8>



```python
z = y * y * 3
out = z.mean()
print(z, out)
```

    tensor([[27., 27.],
            [27., 27.]], grad_fn=<MulBackward0>) tensor(27., grad_fn=<MeanBackward1>)


`.requires_grad_( ... )`会改变张量的`requires_grad`标记。如果没有提供相应的参数,则输入的标记默认为`False`。


```python
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
```

    False
    True
    <SumBackward0 object at 0x7fe543ce5898>


梯度：我们现在后向传播，因为输出包含了一个标量，`out.backward()`等同于`out.backward(torch.tensor(1.))`。


```python
out.backward()
```

打印梯度`d(out)/dx`


```python
print(x.grad)
```

    tensor([[4.5000, 4.5000],
            [4.5000, 4.5000]])


现在让我们看一个雅可比向量积的例子：


```python
x = torch.randn(3, requires_grad=True)
y = x ** 2
while y.data.norm() < 1000:
    y = y ** 2


print(y)
```

    tensor([1.5124e-03, 2.8264e-03, 2.7716e+04], grad_fn=<PowBackward0>)


现在在这种情况下，y 不再是一个标量。torch.autograd 不能够直接计算整个雅可比，但是如果我们只想要雅可比向量积，只需要简单的传递向量给 backward 作为参数。


```python
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)
```

    tensor([-5.9286e-03,  1.0865e-01,  6.4423e+01])


你可以通过将代码包裹在 with torch.no_grad()，来停止对从跟踪历史中 的 .requires_grad=True 的张量自动求导。


```python
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
```

    True
    True
    False



```python

```


```python

```


```python

```
