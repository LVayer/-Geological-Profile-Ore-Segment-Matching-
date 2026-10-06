"""气球总体轮廓实验的兼容入口。

显式点列求解器在复杂接触处容易发生节点冻结或自交，现统一调用经过包含性、
连通性和单调收缩审计的隐式弹性膜实现。
"""
from experimental_implicit_balloon import main


if __name__ == "__main__":
    main()
