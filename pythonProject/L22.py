# s = "ab"
# t = "a"
# res = {}
# s_len = len(s)
# t_len = len(t)
# for i in s:
#     if i in res:
#         res[i] = res[i] + 1
#     else:
#         res[i] = 1
# for j in t:
#     if j in res and t.count(j) == res[j]:
#         continue
#     else:
#         print(False)
# print(True)

# 避免KeyError异常
# from collections import defaultdict
# fruit = ['apple','banana','apple','cherry']
# count = defaultdict(int)
# for fruit in fruit:
#     count[fruit] +=1
# print(count)

# 高效计数
# from collections import Counter
# a = Counter("apple")
# print(a)

# print("hello world")

