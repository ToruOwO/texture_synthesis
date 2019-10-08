import numpy as np
import scipy.stats as st
from scipy import ndimage
import math
import cv2
import random
import math
import matplotlib.pyplot as plt

def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float) / info.max # Divide all values by the largest possible value in the datatype

    def im2col_sliding_strided(img, w, stepsize=1):
    # Parameters
    m, n = img.shape
    col_extent = n - w + 1
    row_extent = m - w + 1

    # Get Starting block indices
    start_idx = np.arange(w)[:, None]*n + np.arange(w)

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:,None]*n + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    return np.take (img, start_idx.ravel()[:,None] + offset_idx.ravel()[::stepsize])

def find_matches(template, sample):
    """
    For a given pixel in the target image, returns a
    list of possible candidate matches in the source
    texture along with their corresponding SSD errors.
    
    Inputs:
      template - w*w*3 image template associated with
        a pixel of the target image.
      sample - the source texture image.
      
    Returns:
      best_matches - a list of possible candidate matches.
      error - SSD errors of the candidate matches.
    """
    epsilon = 0.1
    
    w = template.shape[0]
    
    # validMask is a square mask of width w that is 1 where template is filled
    valid_mask = np.zeros_like(template)
    valid_mask = np.where(np.isnan(template), 0, 1)
    
    def G(x):
        """
        G is a 2D zero-mean Gaussian with variance w/6.4 sampled 
        on a w*w grid centered about its mean.
        w is the size of the search window.
        """
        sigma = w / 6.4
        x = np.linspace(-sigma, sigma, w+1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        
        # normalize mask such that its elements sum to 1
        return kern2d / kern2d.sum()
    
    # (5, 5)
    mask = G(valid_mask)
    
    # (25, 1)
    mask_vec = np.reshape(mask, (-1, 1)) / np.sum(mask)

    # partition sample to blocks (represented by column vectors)
    
    # (25, n_blocks)
    r_sample = im2col_sliding_strided(sample[:, :, 0], w)
    g_sample = im2col_sliding_strided(sample[:, :, 1], w)
    b_sample = im2col_sliding_strided(sample[:, :, 2], w)
    
    n_blocks = r_sample.shape[-1]   # (m-w+1)*(n-w+1)
    
    # vectorized code that calcualtes SSD(template,sample)*mask for all
    # patches
    
    # (25, 1)
    r_temp = np.reshape(template[:, :, 0], (w*w, 1))
    g_temp = np.reshape(template[:, :, 1], (w*w, 1))
    b_temp = np.reshape(template[:, :, 2], (w*w, 1))
    
    # (25, n_blocks)
    r_temp = np.tile(r_temp, (1, n_blocks))
    g_temp = np.tile(g_temp, (1, n_blocks))
    b_temp = np.tile(b_temp, (1, n_blocks))
    
    r_dist = mask_vec * (r_temp - r_sample)**2
    g_dist = mask_vec * (g_temp - g_sample)**2
    b_dist = mask_vec * (b_temp - b_sample)**2
    
    # (25, n_blocks) -> (n_blocks)
    ssd = np.nansum(np.nansum([r_dist, g_dist, b_dist], axis=0), axis=0)

    # accept all pixel locations whose SSD error values are less than the 
    # minimum SSD value times (1 + ε)
    matches = np.nonzero(ssd)
    errors = ssd[matches]
    min_error = np.min(errors)
    idx = np.where(errors < min_error*(1+epsilon))
    
    best_matches = [matches[0][i] for i in idx[0]]
    errors = [errors[i] for i in idx[0]]
    
    return best_matches, errors

def synth_texture(sample, w, s, fixed_seed=True):
    """Texture Synthesis by Non-parameteric Sampling / Efros and Leung."""
    
    # normalize pixel intensity
    sample = im2double(sample)
    m, n = sample.shape[:-1]
    nrows = m - w + 1
    ncols = n - w + 1
    
    # height and width of the target image
    theight = s[0]
    twidth = s[1]
    
    seed_size = 3
    [sheight, swidth, nChannels] = sample.shape
    max_error = 0.3
    
    synth_im = np.full((theight, twidth, nChannels), np.nan)

    ### Initialization: pick a random 3x3 patch from sample and place in the middle of the synthesized image.
    ### Just for convenience, keep some space (SEED_SIZE) from the boundary
    
    if fixed_seed:
        i0=31
        j0=3
    else:
        i0 = round(seed_size + np.random.uniform(0,1) * (sheight - 2 * seed_size))
        j0 = round(seed_size + np.random.uniform(0,1) * (swidth - 2 * seed_size))
    
    c = [round(.5 * x) for x in s]   # middle indices of synth_im
    synth_im[c[0]: c[0] + seed_size , c[1]: c[1] + seed_size ,:] = sample[i0: i0 + seed_size , j0: j0 + seed_size,:]
    
    ### bitmap indicating filled pixels in target image synth_im
    filled = np.zeros(s)
    filled[c[0]: c[0] + seed_size , c[1]: c[1] + seed_size] = 1
    n_filled = int(np.sum(filled))
    n_pixels = s[0]*s[1]

    ### Main Loop
    progress = 0
    while(n_filled < n_pixels):
        # report progress
        if n_filled // 1000 > progress:
            print('{}/{} complete'.format(n_filled, n_pixels) )
            progress += 1

        # dilate current boundary, find the next round of un-filled pixels
        dilated = ndimage.binary_dilation(filled).astype(filled.dtype) - filled
        pixels = np.nonzero(dilated)
        
        ### TODO:
        # permute (just to insert some random noise, not a must, but recommended)
        
        for ii in range(len(pixels[0])):
            # get indices of the next unfilled pixel
            i, j = pixels[0][ii], pixels[1][ii]
        
            # place window at the center
            # shape of template (w, w, 3)
            padded_im = np.pad(synth_im,
                       ((w//2, w//2), (w//2, w//2), (0, 0)),
                       mode='constant',
                       constant_values=np.nan)
            template = padded_im[i:i+w, j:j+w, :]
            
            best_matches, errors = find_matches(template, sample)
            
            # randomly sample from best matches
            if len(errors) ==1:
                r = 0
            else:
                r = np.random.randint(0,  len(best_matches)-1)
            best_match = best_matches[r]
            
            # also check that the error of the randomly selected match is below max_error
            if errors[r] < max_error:
                s_row, s_col = np.unravel_index(best_match, (nrows, ncols))
                synth_im[i, j, :] = sample[s_row+w//2, s_col+w//2, :]
                n_filled += 1
                filled[i, j] = 1
                # print("bm", s_row, s_col, i, j)
        
        # increase error bound if pixel is unfilled
        if filled[i, j] == 0:
            max_error *= 1.1

    return synth_im


if __name__ == '__main__':
	source = cv2.imread('rings.jpg')
	w = 5
	target = synth_texture(source, w, [100, 100])

	plt.imshow(target)
	plt.title('w =' + str(w))
	plt.show()