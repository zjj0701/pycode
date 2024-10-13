





def combine_num(*args):
    yield from args

li = [x for x in range(100)]
s = set(x for x in range(101,200))

print(combine_num((li, s)))