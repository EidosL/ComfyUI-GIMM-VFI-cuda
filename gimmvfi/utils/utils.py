import numpy as np
import torch
import torch.nn.functional as F

class InputPadder:
    """ Pads images such that dimensions are divisible by divisor """
    def __init__(self, dims, divisor=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(I0, I1, flow):
    
    I0_warped = warp(I0, flow)
    I1_warped = warp(I1, -flow)
    
    return I0_warped, I1_warped

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)
    
    if x.dtype == torch.float16:
        vgrid = vgrid.half()
        
    output = F.grid_sample(x, vgrid, padding_mode='ones', align_corners=True)
    # mask = torch.ones(x.size()).cuda()
    # mask = F.grid_sample(mask, vgrid, padding_mode='zeros')

    # mask[mask<0.9999] = 0
    # mask[mask>0] = 1
    
    return output #* mask

# --------------------------------------------------------------------------- #
# The missing function 'flow_to_image' and its dependencies                   #
# Source: Adapted from RAFT visualization utilities                           #
# --------------------------------------------------------------------------- #

def flow_to_image(flow, convert_to_bgr=False):
    """
    Convert flow map to a color image.
    
    Args:
        flow (ndarray): Optical flow map in the shape of [H, W, 2].
        convert_to_bgr (bool): Whether to convert the image to BGR format.
        
    Returns:
        ndarray: Color-coded flow image in the shape of [H, W, 3].
    """
    UNKNOWN_FLOW_THRESH = 1e7
    SMALL_FLOW_THRESH = 1e-6
    
    flow_magnitude, flow_angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # A simple way to suppress noise in low-flow areas
    flow_magnitude[flow_magnitude < SMALL_FLOW_THRESH] = 0

    # Create HSV image
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = flow_angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(flow_magnitude * 4, 255)
    
    # Handle unknown flow
    hsv[flow_magnitude > UNKNOWN_FLOW_THRESH] = 0
    
    # Convert HSV to RGB (or BGR)
    if convert_to_bgr:
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
    return img

try:
    import cv2
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "flow_viz_libs"))
    import cv2
