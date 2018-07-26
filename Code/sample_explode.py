import skimage
import io,os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageChops 

def move(img_path, off_x, off_y): 
	img = Image.open(img_path)
	offset = ImageChops.offset(img, off_x, off_y )
	return offset

def flip(img_path):  
	img = Image.open(img_path)
	filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
	# filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
	return filp_img

def aj_contrast(img_path): #Contrast
	image = skimage.io.imread(img_path)
	gam= skimage.exposure.adjust_gamma(image, 0.5)
	# skimage.io.imsave(os.path.join(root_path,img_name.split('.')[0] + '_gam.jpg'),gam)
	log= skimage.exposure.adjust_log(image)
	# skimage.io.imsave(os.path.join(root_path,img_name.split('.')[0] + '_log.jpg'),log)
	return gam,log

def rotation(img_path, angle):
	img = Image.open(img_path)
	rotation_img = img.rotate(angle) # Rotate angle
	# rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
	return rotation_img

def randomGaussian(img_path, mean, sigma): 
	image = Image.open(img_path)
	im = np.array(image)
	# Center offset
	#means = 0
	# stdev
	#sigma = 25
	
	r = im[:,:,0].flatten()
	g = im[:,:,1].flatten()
	b = im[:,:,2].flatten()

	for i in range(im.shape[0]*im.shape[1]):

		pr = int(r[i]) + random.gauss(0,sigma)

		pg = int(g[i]) + random.gauss(0,sigma)

		pb = int(b[i]) + random.gauss(0,sigma)

		if(pr < 0):
			pr = 0
		if(pr > 255):
			pr = 255
		if(pg < 0):
			pg = 0
		if(pg > 255):
			pg = 255
		if(pb < 0):
			pb = 0
		if(pb > 255):
			pb = 255
		r[i] = pr
		g[i] = pg
		b[i] = pb
	im[:,:,0] = r.reshape([im.shape[0],im.shape[1]])

	im[:,:,1] = g.reshape([im.shape[0],im.shape[1]])

	im[:,:,2] = b.reshape([im.shape[0],im.shape[1]])

	gaussian_image = Image.fromarray(np.uint8(im))
	return gaussian_image

def randomColor(img_path): 
	image = Image.open(img_path)

	random_factor = np.random.randint(0, 31) / 10.  
	color_image = ImageEnhance.Color(image).enhance(random_factor)  # Saturability

	random_factor = np.random.randint(10, 21) / 10.  
	brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # Brightness
	
	random_factor = np.random.randint(10, 21) / 10.  
	contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # Contrast
	
	random_factor = np.random.randint(0, 31) / 10. 
	return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # Acutance


def case_insensitive_sort(liststring):
		listtemp = [(x.lower(),x) for x in liststring]
		listtemp.sort()
		return [x[1] for x in listtemp]


class ScanFile(object):   
	def __init__(self,directory,prefix=None,postfix=None):  
		self.directory=directory  
		self.prefix=prefix  
		self.postfix=postfix  
		  
	def scan_files(self):    
		
		print "Scan started!"
		files_list=[]    
			
		for dirpath,dirnames,filenames in os.walk(self.directory):   
			''''' 
			dirpath is a string, the path to the directory.   
			dirnames is a list of the names of the subdirectories in dirpath (excluding '.' and '..'). 
			filenames is a list of the names of the non-directory files in dirpath. 
			'''  
			counter = 0
			list=[]
			for special_file in filenames:    
				if self.postfix:    
					special_file.endswith(self.postfix)    
					files_list.append(os.path.join(dirpath,special_file))    
				elif self.prefix:    
					special_file.startswith(self.prefix)  
					files_list.append(os.path.join(dirpath,special_file))    
				else:   
					counter += 1
					list.append(os.path.join(dirpath,special_file)) 
		
		 
			if counter > 2:
				print "Found ",counter," files"
				files_list.extend(list)
		
		files_list=case_insensitive_sort(files_list)   
								  
		return files_list    
	  
	def scan_subdir(self):  
		subdir_list=[]  
		for dirpath,dirnames,files in os.walk(self.directory):  
			subdir_list.append(dirpath)  
		return subdir_list  


def mkdir(path):

	path=path.strip()
	path=path.rstrip("/")
 
	isExists=os.path.exists(path)
 
	if not isExists:
		os.makedirs(path) 
 
		print path+' Sucessfully made the directory!'
		return True
	else:
		print path+' Directory already exists'
		return False



if __name__=="__main__":

	dir = "/home/clarence/Desktop/OceanMiner/Training/0"
	
	split_result = dir.split('/')
	dir_name = split_result[-1]

	print dir_name

	move_dir = os.path.join(dir, "..", dir_name + "_move")
	mkdir(move_dir)
	flip_dir = os.path.join(dir, "..", dir_name + "_flip")
	mkdir(flip_dir)
	contrast_dir = os.path.join(dir,"..", dir_name + "_contrast")
	mkdir(contrast_dir)
	rotation_dir = os.path.join(dir,"..", dir_name + "_rotation")
	mkdir(rotation_dir)
	gaussian_dir = os.path.join(dir,"..", dir_name + "_gaussian")
	mkdir(gaussian_dir)
	#randomcolor_dir = os.path.join(dir,"/randomcolor")
	#mkdir(randomcolor_dir)

	scan = ScanFile(dir)   
	files = scan.scan_files()
	print files

	number = 0
	for file in files:
		if os.path.splitext(file)[1] == '.png' or os.path.splitext(file)[1] == '.jpg':
			
			file_name = os.path.splitext( os.path.split(file)[1] ) [0]

			# Convert
			img_move = move(file, 20, 0)
			img_move.save(os.path.join(move_dir, "_".join((file_name ,"move20.png"))))

			number = number + 1


	print "Processed " + str(number) + " Images"



			
 