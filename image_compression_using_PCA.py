from PIL import Image, ImageOps       # Image Manipulation
import io                             # computing image size (bytes)
from sklearn.decomposition import PCA 
import numpy as np                     
import os
import matplotlib.pyplot as plt       


def get_orig_img_data(img_path,disp=True):
    orig_img=Image.open(img_path)
    img_size_kb=os.stat(img_path).st_size/1024
    ori_pixels=np.array(orig_img.getdata()).reshape(*orig_img.size,-1)
    img_dimension=ori_pixels.shape
    if disp:
        plt.imshow(orig_img)
        plt.show()
    data_dict={}
    data_dict['img_size_kb']=img_size_kb
    data_dict['img_dimension']=img_dimension
    data_dict['img']=ori_pixels
    return data_dict

def pca_compose(img_path):
    orig_img=Image.open(img_path)
    img=np.array(orig_img.getdata())
    img=img.reshape(*orig_img.size,-1)
    pca_channel={}
    img_t=np.transpose(img)
    for i in range(img.shape[-1]):
        channel=img_t[i].reshape(*img.shape[:2])
        pca=PCA(random_state=40)
        fit_pca=pca.fit_transform(channel)
        pca_channel[i]=(pca,fit_pca)
    
    return pca_channel

def pca_transform(pca_channel,n_components):
    temp_res=[]
    for channel in range(len(pca_channel)):
        pca,fit_pca=pca_channel[channel]
        pca_pixel=fit_pca[:, :n_components]
        pca_comp=pca.components_[:n_components,:]
        compressed_pixel=np.dot(pca_pixel,pca_comp)+pca.mean_
        temp_res.append(compressed_pixel)
    compressed_img=np.transpose(temp_res)
    compressed_img=np.array(compressed_img,dtype=np.uint8)

    return compressed_img

def processed_img_data(img,pca_channel,n_components,disp=True):  #img = array
    img_size_kb = imageSize(img)
    img_dimension = img.shape
    if disp:
        plt.imshow(img)
        plt.show()
    data_dict = {}
    data_dict['img_size_kb'] = img_size_kb
    data_dict['img_dimension'] = img_dimension
    data_dict['img'] = img
    # var_exp = explained_var_n(pca_channel,n_components)
    # data_dict['Variance Explained'] = var_exp
    return data_dict

def imageSize(img):
    buf = io.BytesIO()
    image = Image.fromarray(np.uint8(img))
    image.save(buf, format='png')
    byte_im = buf.tell()
    return byte_im/1024

if __name__ == "__main__":
    img_path = "N:\Machine learning\Algorithms\ori.png";n_components = 50
    data_dict_ori = get_orig_img_data(img_path)
    print('Original Image Data')
    print('Original Image size (kB)',data_dict_ori['img_size_kb'])
    print('Original Image dimension',data_dict_ori['img_dimension'])
    pca_channel = pca_compose(img_path)
    compressed_image = pca_transform(pca_channel,n_components=n_components)
    # var_exp = explained_var_n(pca_channel,n_components=n_components)
    data_dict_comp_img = processed_img_data(compressed_image,pca_channel,n_components)
    print('Compressed Image Data')
    print('Compressed Image size (kB)',data_dict_comp_img['img_size_kb'])
    print('Compressed Image dimension',data_dict_comp_img['img_dimension'])    
    # print('Variance Explained by PCA in the compressed image',data_dict_comp_img['Variance Explained'])
    # variance_added_pc(pca_channel)
    # plot_s_pc(pca_channel)
    fig = plt.figure()
    # plot_v_pc(pca_channel)
# orig_img_data('N:\Machine learning\Algorithms\ori.png')