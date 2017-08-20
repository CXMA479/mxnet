def topleft_crop(img,tl_xy,HW):
  #logging.debug('imgshape:%s, tl_xy:%s, HW:%s'%(str(img.shape),str(tl_xy),str(HW)))
  assert len(img.shape)==4,'expected dim=4, got %d'%len(img.shape)
  t_h,t_w=HW[0],HW[1]
  h,w=img.shape[-2:]
  assert h>0 and w>0,'invalid (h,w): %s'%(h,w)
  if h==t_h and w==t_w:
    return img
  assert h>=t_h and w>=t_w,'not support resize in crop, (%d,%d) vs (%d,%d)'%(h, w,t_h, t_w)
  h_offset=tl_xy[1] if tl_xy[1]+t_h <= h else tl_xy[1]-1
  w_offset=tl_xy[0] if tl_xy[0]+t_w <= w else tl_xy[0]-1
  assert h_offset>=0 and w_offset >=0
  return img[:,:,h_offset:h_offset+t_h,w_offset:w_offset+t_w]

