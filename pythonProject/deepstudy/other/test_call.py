class Person:
    def __call__(self, name):
        print("__call__"+"hello"+name)
    def hello(self, name):
        print("hello"+name)

p =Person()
p.hello("张三")
p("jb")