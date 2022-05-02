import numpy as np

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

# Load and crop data of coarse compliance 32x32
d1 = np.load('dataset/coarse_compliance.npy')
ori_dim = 32
np.random.seed(0)
perm = np.random.permutation(d1.shape[0])
d1=d1[perm]
d = d1[:].reshape(-1,ori_dim,ori_dim)
print('before fragment:',d.shape)
frag_scale = 8
frag_dim = ori_dim//frag_scale
ori_frag = np.zeros((1,frag_dim,frag_dim))
for i in range(d.shape[0]):
    oritmp = fragmentize_overlap_more(d[i].reshape(ori_dim,ori_dim),frag_dim,frag_dim)
    ori_frag = np.concatenate((ori_frag,oritmp.reshape(-1,frag_dim,frag_dim)),axis=0)
ori_frag = ori_frag[1:].reshape(-1,frag_dim,frag_dim)
print(ori_frag.shape)
np.save('dataset/coarse_compliance_fragmented.npy',ori_frag)

# Load and crop data of fine compliance 512x512
d1 = np.load('dataset/fine_compliance.npy')
ori_dim = 512
d1=d1[perm]
d = d1[:].reshape(-1,ori_dim,ori_dim)
frag_scale = 8
frag_dim = ori_dim//frag_scale
ori_frag = np.zeros((1,frag_dim,frag_dim))
for i in range(d.shape[0]):
    oritmp = fragmentize_overlap_more(d[i].reshape(ori_dim,ori_dim),frag_dim,frag_dim)
    ori_frag = np.concatenate((ori_frag,oritmp.reshape(-1,frag_dim,frag_dim)),axis=0)
ori_frag = ori_frag[1:].reshape(-1,frag_dim,frag_dim)
print(ori_frag.shape)
np.save('dataset/fine_compliance_fragmented.npy',ori_frag)

# Load and crop data of coarse density 32x32
d1 = np.load('dataset/coarse_density.npy')
ori_dim = 32
d1=d1[perm]
d = d1[:].reshape(-1,ori_dim,ori_dim)
print('before fragment:',d.shape)
frag_scale = 8
frag_dim = ori_dim//frag_scale
ori_frag = np.zeros((1,frag_dim,frag_dim))
for i in range(d.shape[0]):
    oritmp = fragmentize_overlap_more(d[i].reshape(ori_dim,ori_dim),frag_dim,frag_dim)
    ori_frag = np.concatenate((ori_frag,oritmp.reshape(-1,frag_dim,frag_dim)),axis=0)
ori_frag = ori_frag[1:].reshape(-1,frag_dim,frag_dim)
print(ori_frag.shape)
np.save('dataset/coarse_density_fragmented.npy',ori_frag)

# Load and crop data of fine density 512x512
d1 = np.load('dataset/fine_density.npy')
ori_dim = 512
d1=d1[perm]
d = d1[:].reshape(-1,ori_dim,ori_dim)
frag_scale = 8
frag_dim = ori_dim//frag_scale
ori_frag = np.zeros((1,frag_dim,frag_dim))
for i in range(d.shape[0]):
    oritmp = fragmentize_overlap_more(d[i].reshape(ori_dim,ori_dim),frag_dim,frag_dim)
    ori_frag = np.concatenate((ori_frag,oritmp.reshape(-1,frag_dim,frag_dim)),axis=0)
ori_frag = ori_frag[1:].reshape(-1,frag_dim,frag_dim)
print(ori_frag.shape)
np.save('dataset/fine_density_fragmented.npy',ori_frag)
