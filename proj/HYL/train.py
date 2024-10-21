import os


def makedir(dir):
    os.makedirs(dir, exist_ok=True)


if __name__ == '__main__':
    print("一个简单的基于主动学习环肽Pnear训练流程")
    print("第一步：设置工作目录")
    PROJECTDIR = '/mnt/d/CycPepAL';
    makedir(PROJECTDIR)

