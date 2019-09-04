import numpy as np; 
import matplotlib.pyplot as plt;
from PIL import Image


img = Image.open('../../img/GolfOverhead.png'); 

speedImg = Image.open('../../img/golfSpeed.png'); 

#Notes:
#lack of R seems promising
#green defintely indicates the fairway and green

# imgn = np.asarray(img); 
# plt.imshow(imgn[:,:,1],cmap='gray'); 
# plt.show();

speedImg = np.asarray(speedImg,dtype=np.int32); 

print(speedImg[0,0,:]);
print(speedImg[0,0,1]-speedImg[0,0,0])
tmp = (speedImg[:,:,1]) - speedImg[:,:,0]; 


fig,axarr = plt.subplots(1,3); 
axarr[0].imshow(speedImg[:,:,0],cmap='gray'); 
axarr[1].imshow(speedImg[:,:,1],cmap='gray'); 
axarr[2].imshow(tmp,cmap='gray'); 
plt.show();


