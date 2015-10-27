from PIL import Image

img = Image.new('L', (84, 84))
f = open("test.txt", "r")

line = f.readlines()[0]
# print line
values = line.strip().split(" ")
# print len(values)
#img.putdata(values)
#img.show()
for r in range(84):
    for c in range(84):
        # img[r, c] = values[r * 84 + c]
        img.putpixel((r, c), int(values[r * 84 + c]))
    
img.save("out.png")
