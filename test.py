#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
from model import build_model

if __name__ == "__main__":
    model = build_model([320,320,3])
    model.summary()

    for i,layer in enumerate(model.layers):
        print(i,layer.name)