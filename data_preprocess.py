from preprocess import Preprocesses
import config

input_datadir = './training_dir'
output_datadir = './faces_dir'

obj=Preprocesses(input_datadir,output_datadir)
nrof_images_total,nrof_successfully_aligned=obj.collect_data()

<<<<<<< HEAD
=======


>>>>>>> origin/master
print('Total number of images: %d' % nrof_images_total)
print('Number of successfully aligned images: %d' % nrof_successfully_aligned)




