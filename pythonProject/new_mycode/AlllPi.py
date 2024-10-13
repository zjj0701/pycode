from typing import List


def permute(nums: List[int]) -> List[List[int]]:
    breakpoint()
    def backtrace(first=0):
        if first == n:
            res.append(nums[:])
        for i in range(first, n):
            nums[first], nums[i] = nums[i], nums[first]
            backtrace(first + 1)
            nums[first], nums[i] = nums[i], nums[first]

    n = len(nums)
    res = []
    backtrace()
    return res


# print(permute([1, 2, 3]))


def judgeinsame(str1: str, str2: str) -> bool:
    breakpoint()
    s1 = []
    s2 = []
    for i in str1:
        s1.append(i)
    s1.sort()
    for i in str2:
        s2.append(i)
    for i in range(len(s2)):
        tmp = s2[i:len(s1) + i]
        tmp.sort()
        if tmp == s1:
            return True
    return False


print(judgeinsame("ab", "eidbaooo"))
