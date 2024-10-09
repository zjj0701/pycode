height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
max_res = 0
i = 0
j = len(height) - 1
while i <= j:
    if height[i] < height[j]:
        max_res = max(max_res, (j - i) * height[i])
        i += 1
    elif height[i] > height[j]:
        max_res = max(max_res, (j - i) * height[j])
        j -= 1
    else:
        max_res = max(max_res, (j - i) * height[i])
        i += 1
print(max_res)
