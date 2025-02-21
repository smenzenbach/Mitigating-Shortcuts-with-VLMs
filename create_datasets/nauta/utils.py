import numpy as np
import cv2
import scipy.stats as st
import glob
import time
import os
import math
from numpy.random import randint
from numpy import random
import argparse


def gauss_kernel(size=21, sigma=3, inchannels=3, outchannels=3):
    interval = (2 * sigma + 1.0) / size
    x = np.linspace(-sigma-interval/2,sigma+interval/2,size+1)
    ker1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(ker1d, ker1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((1, 1, size, size))
    out_filter = np.tile(out_filter, [outchannels, inchannels, 1, 1])
    return out_filter


def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)

        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)

        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)

        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask


def generate_rect_mask(im_size, mask_size, margin=8, rand_mask=True):
    mask = np.zeros((im_size[0], im_size[1])).astype(np.float32)
    if rand_mask:
        sz0, sz1 = mask_size[0], mask_size[1]
        of0 = np.random.randint(margin, im_size[0] - sz0 - margin)
        of1 = np.random.randint(margin, im_size[1] - sz1 - margin)
    else:
        sz0, sz1 = mask_size[0], mask_size[1]
        of0 = (im_size[0] - sz0) // 2
        of1 = (im_size[1] - sz1) // 2
    mask[of0:of0+sz0, of1:of1+sz1] = 1
    mask = np.expand_dims(mask, axis=0)
    mask = np.expand_dims(mask, axis=0)
    rect = np.array([[of0, sz0, of1, sz1]], dtype=int)
    return mask, rect


def generate_stroke_mask(im_size, parts=10, maxVertex=20, maxLength=100, maxBrushWidth=24, maxAngle=360):
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)
    mask = np.transpose(mask, [2, 0, 1])
    mask = np.expand_dims(mask, 0)
    return mask


def generate_mask(type, im_size, mask_size):
    if type == 'rect':
        return generate_rect_mask(im_size, mask_size)
    elif type == 'ellipse':
        return generate_ellipse_mask(im_size, mask_size), None
    else:
        return generate_stroke_mask(im_size), None


def getLatest(folder_path):
    files = glob.glob(folder_path)
    file_times = list(map(lambda x: time.ctime(os.path.getctime(x)), files))
    return files[sorted(range(len(file_times)), key=lambda x: file_times[x])[-1]]




def find_angle(pos1, pos2, ret_type = 'deg'):
    # Find the angle between two pixel points, pos1 and pos2.
    angle_rads = math.atan2(pos2[1] - pos1[1], pos2[0] - pos1[1])
    
    if ret_type == 'rads':
        return angle_rads
    elif ret_type == 'deg':
        return math.degrees(angle_rads)                                     # Convert from radians to degrees.


def sample_centre_pts(n, imsize, xlimits=(50,250), ylimits=(50,250)):
    # Function to generate random sample of points for the centres of the elliptical masks.
    pts = np.empty((n,2))                                                   # Empty array to hold the final points
    
    count=0
    while count < n:
        sample = randint(0, imsize[0], (n,2))[0]                            # Assumes im_size is symmetric

        # Check the point is in the valid region.
        is_valid = (sample[0] < xlimits[0]) | (sample[0] > xlimits[1]) |     \
                (sample[1] < ylimits[0]) | (sample[1] > ylimits[1])
        
        if is_valid:                                                        # Only take the point if it's within the valid region.
            pts[count] = sample
            count += 1

    return pts

def generate_ellipse_mask(imsize, mask_size, seed=None):
    im_centre = (int(imsize[0]/2), int(imsize[1]/2))
    x_bounds =  (int(0.1*imsize[0]), int(imsize[0] - 0.1*imsize[0]))        # Bounds for the valid region of mask centres.
    y_bounds =  (int(0.1*imsize[1]), int(imsize[1] - 0.1*imsize[1]))
    
    if seed is not None:
      random.seed(seed)   # Set seed for repeatability

    n = 1 + random.binomial(1, 0.3)                                         # The number of masks per image either 1 (70% of the time) or 2 (30% of the time) 
    centre_pts = sample_centre_pts(n, imsize, x_bounds, y_bounds)           # Get a random sample for the mask centres.
    
    startAngle = 0.0
    endAngle = 360.0                                                        # Draw full ellipses (although part may fall outside the image)
        
    mask = np.zeros((imsize[0], imsize[1], 1), np.float32)                  # Create blank canvas for the mask.

    for pt in centre_pts:
        size = abs(int(random.normal(mask_size, mask_size/5.0)))            # Randomness introduced in the mask size. 
        ratio = 2*random.random(1) + 1                                      # Ratio between length and width. Sample from Unif(1,3).
        
        centrex = int(pt[0])
        centrey = int(pt[1])
        
        angle = find_angle(im_centre, (centrex, centrey))                   # Get the angle between the centre of the image and the mask centre.
        angle = int(angle + random.normal(0.0, 5.0))                        # Base the angle of rotation on the above angle.
        
        mask = cv2.ellipse(mask, (centrex,centrey), (size, int(size*ratio)), 
                           angle, startAngle, endAngle, 
                           color=1, thickness=-1)                         # Insert a ellipse with the parameters defined above.

    mask = np.minimum(mask, 1.0)                                          # This may be redundant.
    mask = np.transpose(mask, [2, 0, 1])                                  # bring the 'channel' axis to the first axis.
    mask = np.expand_dims(mask, 0)                                        # Add in extra axis at axis=0 - resulting shape (1, 1, )

    return mask

class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--dataset', type=str, default='isic',
                                 help='dataset of the experiment.')
        self.parser.add_argument('--data_file', type=str, default=os.path.join('..', 'modified_data', 'benign_files.txt'), help='the file storing training image paths')
        self.parser.add_argument('--mask_dir', type=str, default=os.path.join('..', 'modified_data', 'dilated-masks'), help='the directory storing mask files')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2')
        self.parser.add_argument('--checkpoint_dir', type=str, default=os.path.join('..', 'models', 'inpainting_gmcnn'), help='models are saved here')
        self.parser.add_argument('--load_model_dir', type=str, default=os.path.join('..', 'models', 'pretrained_gmcnn'), help='pretrained models are given here')
        self.parser.add_argument('--phase', type=str, default='train')

        # input/output sizes
        self.parser.add_argument('--batch_size', type=int, default=8, help='input batch size')

        # for setting inputs
        self.parser.add_argument('--random_crop', type=int, default=1,
                                 help='using random crop to process input image when '
                                      'the required size is smaller than the given size')
        self.parser.add_argument('--random_mask', type=int, default=1)
        self.parser.add_argument('--mask_type', type=str, default='ellipse')
        self.parser.add_argument('--pretrain_network', type=int, default=0)
        self.parser.add_argument('--lambda_adv', type=float, default=1e-3)
        self.parser.add_argument('--lambda_rec', type=float, default=1.4)
        self.parser.add_argument('--lambda_ae', type=float, default=1.2)
        self.parser.add_argument('--lambda_mrf', type=float, default=0.05)
        self.parser.add_argument('--lambda_gp', type=float, default=10)
        self.parser.add_argument('--random_seed', type=bool, default=False)
        self.parser.add_argument('--padding', type=str, default='SAME')
        self.parser.add_argument('--D_max_iters', type=int, default=5)
        self.parser.add_argument('--lr', type=float, default=1e-5, help='learning rate for training')

        self.parser.add_argument('--train_spe', type=int, default=650)
        self.parser.add_argument('--epochs', type=int, default=40)
        self.parser.add_argument('--viz_steps', type=int, default=25)
        self.parser.add_argument('--spectral_norm', type=int, default=1)

        self.parser.add_argument('--img_shapes', type=str, default='224,224,3',
                                 help='given shape parameters: h,w,c or h,w')
        self.parser.add_argument('--mask_shapes', type=str, default='40',
                                 help='given mask parameters: h,w   or if mask_type==ellipse then this should be a number to represent the width of the ellipse')
        self.parser.add_argument('--max_delta_shapes', type=str, default='32,32')
        self.parser.add_argument('--margins', type=str, default='0,0')


        # for generator
        self.parser.add_argument('--g_cnum', type=int, default=32,
                                 help='# of generator filters in first conv layer')
        self.parser.add_argument('--d_cnum', type=int, default=64,
                                 help='# of discriminator filters in first conv layer')

        # for id-mrf computation
        self.parser.add_argument('--vgg19_path', type=str, default='vgg19_weights/imagenet-vgg-verydeep-19.mat')
        # for instance-wise features
        self.initialized = True

    def parse(self, args=[]):
        if not self.initialized:
            self.initialize()

        if isinstance(args, dict):                    # If args is supplied as a dict, flatten to a list.
            args = [item for pair in args.items() for item in pair]
        elif not isinstance(args, list):              # Otherwise, it should be a list.
            raise('args should be a dict or a list.')

        self.opt = self.parser.parse_args(args=args)

        self.opt.dataset_path = self.opt.data_file

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(str(id))

        assert self.opt.random_crop in [0, 1]
        self.opt.random_crop = True if self.opt.random_crop == 1 else False

        assert self.opt.random_mask in [0, 1]
        self.opt.random_mask = True if self.opt.random_mask == 1 else False

        assert self.opt.pretrain_network in [0, 1]
        self.opt.pretrain_network = True if self.opt.pretrain_network == 1 else False

        assert self.opt.spectral_norm in [0, 1]
        self.opt.spectral_norm = True if self.opt.spectral_norm == 1 else False

        assert self.opt.padding in ['SAME', 'MIRROR']

        assert self.opt.mask_type in ['rect', 'stroke', 'ellipse']

        str_img_shapes = self.opt.img_shapes.split(',')
        self.opt.img_shapes = [int(x) for x in str_img_shapes]

        if self.opt.mask_type == 'ellipse':
          self.opt.mask_shapes = int(self.opt.mask_shapes)
        else:
          str_mask_shapes = self.opt.mask_shapes.split(',')
          self.opt.mask_shapes = [int(x) for x in str_mask_shapes]

        str_max_delta_shapes = self.opt.max_delta_shapes.split(',')
        self.opt.max_delta_shapes = [int(x) for x in str_max_delta_shapes]

        str_margins = self.opt.margins.split(',')
        self.opt.margins = [int(x) for x in str_margins]

        # model name and date
        self.opt.date_str = time.strftime('%Y%m%d-%H%M%S')
        self.opt.model_name = 'GMCNN'
        self.opt.model_folder = self.opt.date_str + '_' + self.opt.model_name
        self.opt.model_folder += '_' + self.opt.dataset
        self.opt.model_folder += '_b' + str(self.opt.batch_size)
        self.opt.model_folder += '_s' + str(self.opt.img_shapes[0]) + 'x' + str(self.opt.img_shapes[1])
        self.opt.model_folder += '_gc' + str(self.opt.g_cnum)
        self.opt.model_folder += '_dc' + str(self.opt.d_cnum)

        self.opt.model_folder += '_randmask-' + self.opt.mask_type if self.opt.random_mask else ''
        self.opt.model_folder += '_pretrain' if self.opt.pretrain_network else ''

        if os.path.isdir(self.opt.checkpoint_dir) is False:
            os.mkdir(self.opt.checkpoint_dir)

        self.opt.model_folder = os.path.join(self.opt.checkpoint_dir, self.opt.model_folder)
        if os.path.isdir(self.opt.model_folder) is False:
            os.mkdir(self.opt.model_folder)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(self.opt.gpu_ids)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt