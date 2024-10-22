age = 20

# 1
# msg = '成年' if age > 18 else '未成年'

# 2
# msg = age > 18 and '成年' or '未成年'

# 3
# msg = ('未成年', '成年')[age > 18]

# 4
# msg = {True: "成年", False: "未成年"}[age > 18]

# 5
# msg = '未成年'[age>18:]

# 6
# try:
#     assert age > 18
#     msg = '成年'
# except:
#     msg = '未成年'

# 7
# msg = (["未成年"]*19+["成年"]*120)[age]

# 8
# msg = '成年' if range(age, 18) else '未成年'

# 9
# msg = age // 18 and '成年' or '未成年'

# 10
# msg = (age > 18 and "" or "未") + "成年"
# print(msg)


# 连接列表
a = [1, 2, 3]
b = [5, 6, 7]
c = [*a, *b]
d = a + b

# 合并字典
a = {'a': 1, 'b': 2}
b = {'c': 5, 'd': 6, 'b': 3}

c = {**a, **b}
print(c)
d = a | b
print(d)
