from itertools import count

s = 'ss115599wweertyui'

s_set = set(s)
res = 0
for i in s_set:
    if s.count(i)>1:
        continue
    else:
        res+=1
print(res)
