from matplotlib.widgets import RectangleSelector
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import seaborn as sns
sns.set(style="white", context="talk")
import pyautogui
from skimage import data



def getMask(im):
    def line_select_callback(eclick, erelease):
        'eclick and erelease are the press and release events'
        global coord
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        coord = (x1, y1, x2, y2)

    def toggle_selector(event):
        global coord
        if event.key in ['Q', 'q'] and toggle_selector.RS.active:
            print(' RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)

        if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            print(' RectangleSelector activated.')
            toggle_selector.RS.set_active(True)

        if event.key == 'r':
            coordinates.append(coord)
            c = coordinates[-1]
            p = patches.Rectangle((c[0], c[1]), c[2]-c[0], c[3]-c[1],alpha=0.4)
            p1 = patches.Rectangle((c[0], c[1]), c[2]-c[0], c[3]-c[1],facecolor="white")
            mask_ax.add_patch(p1)
            current_ax.add_patch(p)
            plt.draw()
            coord = np.asarray(c).astype('int')
            cv2.rectangle(maskout,(coord[0],coord[1]),(coord[2],coord[3]),0,-1)
            centers.append(None)

        if event.key == 'c':
            coordinates.append(coord)
            c = coordinates[-1]
            w = c[2]-c[0]
            h = c[3]-c[1]
            p = patches.Ellipse((c[0]+w/2, c[1]+h/2), w, h, angle=0.0, alpha=0.4)
            p1 = patches.Ellipse((c[0]+w/2, c[1]+h/2), w, h, angle=0.0,facecolor="white")
            current_ax.add_patch(p)
            mask_ax.add_patch(p1)
            plt.draw()
            coord = np.asarray(c).astype('int')
            center = ((coord[2]+coord[0])/2,(coord[3]+coord[1])/2)
            angle = 90
            axes = tuple(sorted(((coord[2]-coord[0])/2,(coord[3]-coord[1])/2)))
            cv2.ellipse(maskout,center,axes,angle,0,360,0,-1)
            centers.append(center)

    coordinates = []
    centers = []
    w, h = pyautogui.size()

    fig, ax_arr = plt.subplots(1,2, figsize=(w/96,h/96))
    fig.suptitle('Select mask')
    current_ax = ax_arr[0]
    mask_ax = ax_arr[1]
    current_ax.set_title('Drag on the image, adjust,\n press r, or c to get a rectangular or a circular ROI')
    mask_ax.set_title('Visualise the mask')
    sns.despine(fig=fig, top=True, right=True, left=True, bottom=True)
    current_ax.imshow(im, cmap = 'gray')
    mask = np.zeros_like(im)
    maskout = np.ones_like(im,dtype='uint8')*255
    mask_ax.imshow(mask, cmap = 'gray')

    toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
                                           drawtype='box', useblit=True,
                                           button=[1, 3],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    plt.connect('key_press_event', toggle_selector)
    plt.show()

    return maskout, centers

im = data.camera()

mask,c, cent = getMask(im)
