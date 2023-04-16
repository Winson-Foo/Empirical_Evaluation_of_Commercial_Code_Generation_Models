def calculate_average(numbers):
    total = 0
    count = 0
    for i in numbers:
        total += i
        count += 1
    average = total / count
    return average
    
numbers = [5, 10, 15, 20]
print(calculate_average(numbers))