from generator import Generator

num_iter = 1000
num_classes = 5
cat_mean = [0]*num_classes

for i in range(num_iter):
    try:
        gen = Generator(size=10, num_classes=num_classes)
        grades,admission = gen.generate()
        cat_list = [0]*num_classes
        for cat in admission:
            cat_list[int(cat)] += 1
        for j in range(num_classes):
            cat_mean[j] += cat_list[j]
    except KeyError:
        pass

for i in range(num_classes):
    cat_mean[i]/=(num_iter*gen.size)

print(f"Mean percentage of each class: {cat_mean}")