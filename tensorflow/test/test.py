# python编程中的if __name__ == 'main': 的作用和原理。
# 每个python模块（python文件，也就是此处的test.py和import_test.py）都包含内置的变量__name__。
# 1 如果模块（test.py）被import到其他模块(import_test.py)中，则__name__等于被导入的模块名称（即被import的文件名称）（不包含后缀.py）。
# 2 如果模块(test.py)直接作为脚本执行的话，__name__ 等于“__main__”，进而当文件被直接执行时，__name__ == 'main'结果为真。
print("I'm the first.")
print(__name__)  # __main__
if __name__ == "__main__":
    print("I'm the second.")
