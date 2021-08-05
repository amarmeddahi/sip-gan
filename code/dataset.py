"""
Created on Wed Jun 23 10:42:28 2021

@author: Amar Meddahi (amar.meddahi1[at]gmail.com)
"""

import os
from toolbox import *

# Data Preprocessing
dataset_name ='SIP_INVITE_20000.txt'
dataset_path = os.path.abspath(dataset_name)
nb_packets = 20000 # Number of packets in dataset_name
max_length = 560 # Size in bytes of the largest packet in dataset (cf: Wireshark)
vect_length = max_length*2
packets_matrix = txt_to_matrix(dataset_path, nb_packets, vect_length)

# Encoder
multimap = 2
dir_images = 'images'
os.mkdir(dir_images)
images_path = os.path.abspath(dir_images) + '\\'
create_dataset_images(packets_matrix, multimap, images_path)

# Decoder
text_file_name = 'SIP_INVITE_20000_decoded.txt'
packets_matrix = images_to_matrix(images_path, nb_packets, multimap)
matrix_to_txt(packets_matrix, text_file_name)