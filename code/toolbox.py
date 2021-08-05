# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 10:42:28 2021

@author: Amar Meddahi (amar.meddahi1[at]gmail.com)
"""

from math import floor
import numpy as np
from numpy import asarray
from PIL import Image

def txt_to_matrix(dataset_path, nb_packets, packet_size):
    """
    Convert a text file corresponding to the bytes of n network packets into
    a matrix where each column contains all the information of a single packet.

    Parameters
    ----------
    path_dataset : STR
        DATASET PATH.
    nb_packets : INT
        NUMBER OF PACKETS IN DATASET.
    packet_size : INT
        SIZE OF A SINGLE PACKET.

    Returns
    -------
    Matrix where each column correspond to a single packet.

    """

    rows = 35    # Number of lines corresponding to a packet in the text file
    columns = 16 # Number of columns corresponding to a bytes line
    start = 6    # First bytes postion of a dataset line
    byte_size = 2
    packets_matrix = np.zeros((packet_size, nb_packets))
    dataset = open(dataset_path, encoding='utf-8')  # Open dataset in its directory
    for i in range(nb_packets):
        index = 0
        for j in range(rows):
            dataset.read(start)
            for k in range(columns):
                for l in range(byte_size):
                    temp = dataset.read(1)
                    if temp != ' ' and temp != '\n' and temp != '':
                        packets_matrix[index,i] = int(temp, 16)
                        index += 1
                dataset.read(1)
            next(dataset)
        if i != nb_packets - 1:
            next(dataset)
    dataset.close()
    return packets_matrix

def mapping_function(hexa_value):
    """
    Map an hexadecimal value [0,15] from to a pixel [0,255].

    Parameters
    ----------
    hexa_value : INT
        HEXADECIMAL VALUE TO MAP.

    Returns
    -------
    The hexadecimal value mapped.

    """

    return hexa_value * 16 + 16 // 2

def demapping_function(pixel):
    """
    Map a pixel [0,255] to its corresponding hexadecimal value [0,15].

    Parameters
    ----------
    pixel : INT
        PIXEL FROM THE ENCODED IMAGE.

    Returns
    -------
    The pixel value demapped.

    """

    return floor(((pixel - 8) / 16) + 0.5)

def encoder(packet, image_size, multimap):
    """
    Encode a network packet into an image.

    Parameters
    ----------
    packet : ARRAY
        VECTOR OF BYTES.
    image_size : LIST
        HEIGHT AND WIDTH OF THE IMAGE.
    multimap : INT
        THE MULTIMAPPING PARAMETER.

    Returns
    -------
    A matrix corresponding to the encoded input network packet.

    """

    p_index = 0
    rows = range(0, image_size[0], multimap)
    columns = range(0, image_size[1], multimap)
    image_packet = np.zeros((image_size[0], image_size[1]))
    for i in rows:
        for j in columns:
            image_packet[i:(i + multimap),j:(j + multimap)] = mapping_function(packet[p_index])
            p_index += 1
    return image_packet

def optimal_image_size(multimap):
    """
    Compute the optimal image size given a multimapping parameter.

    Parameters
    ----------
    multimap : INT
        THE MULTIMAPPING PARAMETER.

    Returns
    -------
    The optimal image size.

    """
    if multimap == 2:
        image_size =  [80, 56]
    elif multimap == 4:
        image_size = [160, 112]
    elif multimap == 8:
        image_size = [320, 224]
    elif multimap == 16:
        image_size = [640, 448]
    elif multimap == 32:
        image_size = [1280, 896]
    else:
        image_size = [0, 0]
    return image_size

def create_dataset_images(packets, multimap, save_path):
    """
    Create a dataset of network packets images.

    Parameters
    ----------
    packets : ARRAY
        MATRIX WHERE EACH COLUMN IS A PACKET.
    multimap : INT
        THE MULTIMAPPING PARAMETER.

    Returns
    -------
    None.

    """

    image_size = optimal_image_size(multimap)
    for i in range(packets.shape[1]):
        image = Image.fromarray(encoder(packets[:,i], image_size, multimap).astype(np.uint8))
        image.save(save_path + str(i) + ".png", 'PNG')

def images_to_matrix(images_path, number_images, multimap):
    """
    Convert an dataset of images corresponding to the encoder network packets into
    a matrix where each column contains all the information of a single packet.

    Parameters
    ----------
    images_path : STR
        IMAGES PATH.
    number_images : INT
        NUMBER OF IMAGES.
    multimap : INT
        THE MULTIMAPPING PARAMETER.

    Returns
    -------
    Matrix where each column correspond to a single packet.

    """

    image_size = optimal_image_size(multimap)
    packets_matrix = np.zeros(((image_size[0] * image_size[1]) // (multimap ** 2), number_images))
    rows = range(0, image_size[0], multimap)
    columns = range(0, image_size[1], multimap)
    for i in range(number_images):
        temp = asarray(Image.open(images_path + '/' + str(i) + '.png')).astype(np.int64)
        index = 0
        for j in rows:
            for k in columns:
                pixel = int(temp[j:(j + multimap),k:(k + multimap)].mean())
                packets_matrix[index,i] = demapping_function(pixel)
                index += 1
    return packets_matrix

def matrix_to_txt(packets_matrix, file_name):
    """
    Convert a matrix where each column contains all the information of a single packet
    to a text file.

    Parameters
    ----------
    packets_matrix : ARRAY
        MATRIX OF BYTES.
    file_name : STR
        TEXT FILE NAME.

    Returns
    -------
    None.

    """

    text_file = open(file_name, "w")
    header_length = len(hex(packets_matrix.shape[0])) - 2
    for i in range(packets_matrix.shape[1]):
        bytes_writted = 0
        for j in range(0, packets_matrix.shape[0], 2):
            if not bytes_writted % 15:
                temp = '0'*(header_length - len(hex(bytes_writted)[2:]))
                text_file.write('\n' + temp + hex(bytes_writted)[2:] + '  ')
            text_file.write(hex(int(packets_matrix[j,i]))[-1])
            text_file.write(hex(int(packets_matrix[j+1,i]))[-1])
            text_file.write(' ')
            bytes_writted += 1
        text_file.write('\n')

def images_to_tensor_slices(images_path, number_images):
    """
    Convert a dataset of images to a tensor slices.

    Parameters
    ----------
    images_path : STR
        IMAGES PATH.
    number_images : INT
        NUMBER OF IMAGES.

    Returns
    -------
    A tensor slices.

    """

    image_size = Image.open(images_path + '/0.png').size
    tensor = np.zeros((number_images, image_size[1], image_size[0]))
    for i in range(number_images):
        tensor[i,:,:] = asarray(Image.open(images_path + '/' + str(i) + '.png'))
    return tensor

def stats_error(real_images_path, number_real_images, gen_images_path,
                sample_size, multimap, test_size):
    """
    Compute the byte error between two datasets.

    Parameters
    ----------
    path_1 : STR
        PATH OF THE FIRST DATASET.
    n1 : INT
        NUMBER OF IMAGES in path_1.
    path_2 : TYPE
        PATH OF THE SECOND DATASET.
    n2 : INT
        NUMBER OF IMAGES in path_2.
    multimap : INT
        THE MULTIMAPPING PARAMETER.
    Returns
    -------
    Byte Error.

    """

    real = images_to_matrix(real_images_path, number_real_images, multimap)
    gen = images_to_matrix(gen_images_path, sample_size, multimap)
    byte_error = np.zeros(test_size) # SIP Request-Line : Bytes 42-81

    for i in range(test_size):
        temp = np.zeros((sample_size))
        real_index = np.random.permutation(real.shape[1])
        gen_index = np.random.permutation(gen.shape[1])
        for j in range(sample_size):
            request_line_real = real[84:162,real_index[j]]
            request_line_generated = gen[84:162,gen_index[j]]
            val = (np.absolute(request_line_generated - request_line_real))
            temp[j] = (np.count_nonzero(val == 0))/val.shape[0]
        byte_error[i] = temp.mean()
    return 1 - byte_error.mean()
