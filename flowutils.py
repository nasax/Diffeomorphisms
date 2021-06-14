import numpy as np
import matplotlib.pyplot as plt
import cv2


def readFlo( filename ):

    TAG_FLOAT = 202021.25

    with open(filename,"rb") as f:
        tag = np.fromfile( f, dtype=np.float32, count=1 )
        width = np.fromfile( f, dtype=np.int32, count=1 )
        height = np.fromfile( f, dtype=np.int32, count=1 )

        width = np.asscalar( width )
        height = np.asscalar( height )

        if tag != TAG_FLOAT:
            print('wrong tag (possibly due to big-endian machine?)')

        if width < 1 or width > 99999:
            print('illegal width %d' % width )

        if height < 1 or height > 99999:
            print('illegal height %d' % height )

        nBands = 2

        N=nBands*width*height

        tmp = np.fromfile( f, dtype='f', count=N )
        tmp = np.reshape( tmp, (height,nBands*width) )
        flow = np.zeros( (height, width, 2) )
        flow[:,:,0] = tmp[:,nBands*np.arange(width)]
        flow[:,:,1] = tmp[:,nBands*np.arange(width)+1]

    f.close()

    return flow


def flowToColor( flow, *argv ):

    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWNFLOW = 1e10

    u = flow[:,:,0]
    v = flow[:,:,1]

    maxu = -999
    maxv = -999

    minu = 999
    minv = 999
    maxrad = -1

    idxUnknown = np.logical_or( np.fabs( u ) > UNKNOWN_FLOW_THRESH,  np.fabs( v ) > UNKNOWN_FLOW_THRESH )
    u[idxUnknown] = 0
    v[idxUnknown] = 0

    maxu = np.maximum(maxu, u.max())
    minu = np.minimum(minu, u.min())

    maxv = np.maximum(maxv, v.max())
    minv = np.minimum(minv, v.min())

    #print( "maxu = %f, minu = %f, maxv = %f, minv = %f\n" % (maxu, minu, maxv, minv) )

    rad = np.sqrt(u**2 + v**2)
    maxrad = max(maxrad, rad.max())

    if len(argv) > 0:
        maxFlow = argv[0]
        if maxFlow > 0:
            maxrad = maxFlow

    eps = np.finfo(np.float32).eps
    u /= maxrad + eps
    v /= maxrad + eps

    img = computeColor(u,v)

    IDX = np.expand_dims( idxUnknown, axis=2 )
    IDX = np.tile( IDX, (1,1,3) )
    img[ IDX ] = 0

    return img


def computeColor( u, v ):

    nanIdx = np.logical_or( np.isnan(u), np.isnan(v) )
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = makeColorWheel()
    ncols = colorwheel.shape[0]

    rad = np.sqrt(u**2 + v**2)
    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1 # -1~1 mapped to 1~ncols

    k0 = np.floor(fk).astype(int) - 1 # 1, 2, ..., ncols
    k1 = k0+1
    k1[k1==ncols] = 0

    f = fk - k0 - 1

    img = np.zeros( f.shape + tuple([colorwheel.shape[1]]) )

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255
        col1 = tmp[k1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])  # increase saturation with radius
        col[np.logical_not(idx)] = col[np.logical_not(idx)] * 0.75 # out of range

        img[:,:,i] = np.floor( 255 * col * np.logical_not(nanIdx) )

    return img.astype('uint8')


def makeColorWheel():

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros((ncols, 3))  # r g b

    col = 0
    # RY
    colorwheel[0:RY,0] = 255
    colorwheel[0:RY,1] =  np.floor( 255 * np.arange( 0., RY ) / RY )
    col = col+RY

    # YG
    colorwheel[col:(col+YG), 0] = 255 - np.floor( 255 * np.arange( 0., YG ) / YG )
    colorwheel[col:(col+YG), 1] = 255
    col = col + YG

    # GC
    colorwheel[col:(col+GC), 1] = 255
    colorwheel[col:(col+GC), 2] = np.floor( 255 * np.arange( 0., GC ) / GC )
    col = col + GC

    # CB
    colorwheel[col:(col+CB), 1] = 255 - np.floor( 255 * np.arange( 0., CB ) / CB )
    colorwheel[col:(col+CB), 2] = 255
    col = col + CB

    # BM
    colorwheel[col:(col+BM), 2] = 255
    colorwheel[col:(col+BM), 0] = np.floor( 255 * np.arange( 0., BM ) / BM )
    col = col + BM

    # MR
    colorwheel[col:(col+MR), 2] = 255 - np.floor(255 * np.arange( 0., MR ) / MR)
    colorwheel[col:(col+MR), 0] = 255

    return colorwheel

def colorTest():

    truerange = 1
    height = 151
    width = 151
    range = truerange * 1.04

    s2 = round(height / 2) - 1

    x, y = np.meshgrid(np.arange(width)+1, np.arange(height)+1)

    u = x * range / s2 - range
    v = y * range / s2 - range

    img = computeColor(u / range / np.sqrt(2), v / range / np.sqrt(2))

    img[s2,:,:] = 0
    img[:,s2,:] = 0

    fig = plt.figure(100)
    plt.imshow(img)
    plt.title('optical flow color coding scheme')
#    plt.show()

if __name__ == '__main__':

    #colorTest()

    flow = readFlo( '/Users/sundarg/work/Datasets_Vision/MiddleburyOpticalFlow/other-gt-flow/RubberWhale/flow10.flo')
    fig = plt.figure(1)
    img_flow = flowToColor(flow)
    plt.imshow( img_flow )
    plt.show()

    #flow = ( np.fabs( flow ) < 1e4 ) * flow
    #flow = np.zeros( (100,100,2) )
    #flow[:,:,0] = 1
    #flow[:,:,1] = 0
    #img_flow = flowToColor( flow )
    #plt.imshow( cv2.cvtColor(img_flow, cv2.COLOR_BGR2RGB) )
    #plt.imshow( img_flow )
    #plt.show( )