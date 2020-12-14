import numpy as np
from moviepy.editor import ImageSequenceClip
from IPython.display import Image
import matplotlib.pyplot as plt

def labels2imgs(list_labels, k, H=100, W=100):
    # list_labels: list of ndarray (N,)
    labels = np.asarray(list_labels) # (f, N), f: #frames
    imgs = np.empty((labels.shape[0], H * W, 3))
    
    cmap = plt.get_cmap('tab10')
    for i in range(k):
        imgs[labels == i, :] = cmap(i)[:-1]
    
    imgs = imgs.reshape(-1, H, W, 3)
    return imgs

def plotCluster(imgs, frame=0):
    # imgs: ndarray (f, H, W), f: #frames
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(imgs[frame])
    plt.show()

def saveGIF(imgs, path):
    # imgs: float ndarray (f, H, W), f: #frames
    clip = ImageSequenceClip(list((imgs * 255).astype(np.uint8)), fps=3)
    clip.write_gif(path, fps=3)

def showGIF(path, width=200):
    return Image(path, width=200)

def showEigenSpace(X, k, labels):
    # X: data points in eigenspace, must of shape (N, 2)
    # k: #clusters
    # labels: (N,)
    
    plt.figure(figsize=(10,10))
    
    for i in range(k):
        plt.plot(*zip(*X[labels == i]), 'o', markersize=2)
    
    plt.show()
