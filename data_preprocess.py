from preprocess import Preprocesses
import config

#input_datadir = './training_dir'
#output_datadir = './faces_dir'

#obj=Preprocesses(input_datadir,output_datadir)
obj = Preprocesses(config.TRAINING_DIR, config.FACES_DIR)
nrof_images_total,nrof_successfully_aligned=obj.collect_data()

print('Total number of images: %d' % nrof_images_total)
print('Number of successfully aligned images: %d' % nrof_successfully_aligned)




