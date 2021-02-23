import numpy as np 
from PIL import Image 
import nibabel as nib

def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    array   = ct_scan.get_fdata()
    array   = np.rot90(np.array(array))
    return(array)

def convert(): 
    
    array = read_nii("coronacases.nii")[..., 150]
    print(type(array)) 
    print(array) 
      
      
    data = Image.fromarray(array, 'RGB')
    data.save('corona.png') 
    data.show()
  
if __name__ == "__main__":
	convert() 
