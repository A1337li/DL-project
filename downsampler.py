from PIL import Image

basewidth = 224
img = Image.open('Data_Osteo_Tiles/downsampling_test/test_image.jpg')
wpercent = (basewidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((basewidth,hsize), Image.ANTIALIAS)
img.save('Data_Osteo_Tiles/downsampling_test/test_image.jpg')