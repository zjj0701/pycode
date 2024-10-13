import S

input_str = input("please enter a func: ")

if "fun1" == input_str:
    print(S.fun1())

elif "fun2" == input_str:
    print(S.fun2())

elif "fun3" == input_str:
    print(S.fun3())
else:
    print("func is not found")
# 是否存在这个函数
if hasattr(S, input_str):
    obj = getattr(S, input_str)
    print(obj())
else:
    print("not Found")
