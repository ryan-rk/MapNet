# Module for manipulating arrays
import numpy as np

def fine_arr(inputArr, scale):
    # Making an array bigger by using replicating each element by scale
    outputArr = np.zeros((int(inputArr.shape[0]*scale),int(inputArr.shape[1]*scale)))
    for i in range(inputArr.shape[0]):
        for j in range(inputArr.shape[1]):
            outputArr[i*scale:(i+1)*scale,j*scale:(j+1)*scale] = inputArr[i,j]
    return outputArr

def coarse_arr(inputArr, scale):
    # Making an array smaller by averaging each element
    outputArr = np.zeros((int(inputArr.shape[0]/scale),int(inputArr.shape[1]/scale)))
    for i in range(outputArr.shape[0]):
        for j in range(outputArr.shape[1]):
            outputArr[i,j] = np.mean(inputArr[i*scale:(i*scale)+scale,j*scale:(j*scale)+scale])
    return outputArr

def coarse_arr3d(inputArr, scale):
    # Making an array smaller by averaging each element
    outputArr = np.zeros((int(inputArr.shape[0]/scale),int(inputArr.shape[1]/scale),int(inputArr.shape[2]/scale)))
    for i in range(outputArr.shape[0]):
        for j in range(outputArr.shape[1]):
            for k in range(outputArr.shape[2]):
                outputArr[i,j,k] = np.mean(inputArr[i*scale:(i*scale)+scale,j*scale:(j*scale)+scale,k*scale:(k*scale)+scale])
    return outputArr

def fragmentize(inputArr, frag_width, frag_height):
    # Fragment larger array into smaller parts
    ori_height = inputArr.shape[0]
    ori_width = inputArr.shape[1]
    num_row_array = ori_height//frag_height
    num_col_array = ori_width//frag_width
    outputArr = np.zeros((num_col_array*num_row_array,frag_height,frag_width))
    for i in range(num_row_array):
        for j in range(num_col_array):
            tmp_array = inputArr[i*frag_height:(i+1)*frag_height,j*frag_width:(j+1)*frag_width]
            outputArr[i*num_col_array+j] = tmp_array.reshape(1,frag_height,frag_width)
    return outputArr

def defrag(inputArr, num_frag_row, num_frag_col):
    # Combine fragmented array into larger array
    input_row = inputArr.shape[1]
    input_col = inputArr.shape[2]
    output_row = int(input_row*num_frag_row)
    output_col = int(input_col*num_frag_col)
    outputArr = np.zeros((output_row,output_col))
    for i in range(num_frag_row):
        for j in range(num_frag_col):
            outputArr[i*input_row:(i+1)*input_row,j*input_col:(j+1)*input_col] = inputArr[(i*num_frag_col)+j]
    return outputArr

def fragmentize_overlap(inputArr, frag_width, frag_height):
    # Fragment larger array into smaller parts
    ori_height = inputArr.shape[0]
    ori_width = inputArr.shape[1]
    num_row_array = (ori_height//frag_height)*2 - 1
    num_col_array = (ori_width//frag_width)*2 - 1
    outputArr = np.zeros((num_col_array*num_row_array,frag_height,frag_width))
    for i in range(num_row_array):
        for j in range(num_col_array):
            tmp_array = inputArr[int(i*0.5*frag_height):int(((i*0.5)+1)*frag_height),int(j*0.5*frag_width):int(((j*0.5)+1)*frag_width)]
            outputArr[i*num_col_array+j] = tmp_array.reshape(1,frag_height,frag_width)
    return outputArr

def defrag_overlap(inputArr, num_frag_row, num_frag_col):
    # Combine fragmented array into larger array
    input_row = inputArr.shape[1]
    input_col = inputArr.shape[2]
    output_row = int(input_row*((num_frag_row+1)//2))
    output_col = int(input_col*((num_frag_col+1)//2))
    outputArr = np.zeros((output_row,output_col))
    overlapArr = np.zeros((output_row,output_col))
    for i in range(num_frag_row):
        for j in range(num_frag_col):
            outputArr[int(i*0.5*input_row):int(((i*0.5)+1)*input_row),int(j*0.5*input_col):int(((j*0.5)+1)*input_col)] += inputArr[(i*num_frag_col)+j]
            overlapArr[int(i*0.5*input_row):int(((i*0.5)+1)*input_row),int(j*0.5*input_col):int(((j*0.5)+1)*input_col)] += np.ones((input_row,input_col))
    for m in range(outputArr.shape[0]):
        for n in range(outputArr.shape[1]):
            outputArr[m,n] = outputArr[m,n]/overlapArr[m,n]
    return outputArr

def fragmentize_overlap_more(inputArr, frag_width, frag_height):
    # Fragment larger array into smaller parts
    ori_height = inputArr.shape[0]
    ori_width = inputArr.shape[1]
    num_row_array = (ori_height//frag_height)*2 - 1 + (((ori_height//frag_height)-1)*2)
    num_col_array = (ori_width//frag_width)*2 - 1 + (((ori_width//frag_width)-1)*2)
    outputArr = np.zeros((num_col_array*num_row_array,frag_height,frag_width))
    for i in range(num_row_array):
        for j in range(num_col_array):
            tmp_array = inputArr[int(i*0.25*frag_height):int(((i*0.25)+1)*frag_height),int(j*0.25*frag_width):int(((j*0.25)+1)*frag_width)]
            outputArr[i*num_col_array+j] = tmp_array.reshape(1,frag_height,frag_width)
    return outputArr

def defrag_overlap_more(inputArr, num_frag_row, num_frag_col):
    # Combine fragmented array into larger array
    input_row = inputArr.shape[1]
    input_col = inputArr.shape[2]
    output_row = int((num_frag_row+3)*input_row//4)
    output_col = int((num_frag_col+3)*input_col//4)
    outputArr = np.zeros((output_row,output_col))
    overlapArr = np.zeros((output_row,output_col))
    for i in range(num_frag_row):
        for j in range(num_frag_col):
            outputArr[int(i*0.25*input_row):int(((i*0.25)+1)*input_row),int(j*0.25*input_col):int(((j*0.25)+1)*input_col)] += inputArr[(i*num_frag_col)+j]
            overlapArr[int(i*0.25*input_row):int(((i*0.25)+1)*input_row),int(j*0.25*input_col):int(((j*0.25)+1)*input_col)] += np.ones((input_row,input_col))
    for m in range(outputArr.shape[0]):
        for n in range(outputArr.shape[1]):
            outputArr[m,n] = outputArr[m,n]/overlapArr[m,n]
    return outputArr