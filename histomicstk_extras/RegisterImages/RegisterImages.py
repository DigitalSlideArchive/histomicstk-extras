import copy
import logging
import math
import os
import pprint
import sys
import tempfile

import girder_client
import histomicstk
import large_image
import large_image_converter
import numpy as np
import pystackreg
import rasterio.features
import shapely
import skimage.filters
import skimage.morphology
import skimage.transform
import yaml
from histomicstk.cli.utils import CLIArgumentParser
from histomicstk.preprocessing.color_deconvolution.stain_color_map import \
    stain_color_map

from progress_helper import ProgressHelper


def annotation_to_shapely(annot, reduce=1):
    return shapely.polygons([
        shapely.linearrings([[p[0] / reduce, p[1] / reduce]
                            for p in element['points']])
        for element in annot['annotation']['elements']
        if element['type'] == 'polyline' and element['closed']
    ])


def get_image(ts, sizeX, sizeY, frame, annotID, args, reduce, debug=None, rscale=1):  # noqa
    regionparams = {'format': large_image.constants.TILE_FORMAT_NUMPY}
    try:
        regionparams['frame'] = int(frame)
    except Exception:
        try:
            if 'channelmap' in ts.metadata and 'DAPI' in ts.metadata['channelmap']:
                regionparams['frame'] = (
                    ts.metadata['channelmap']['DAPI'] * ts.metadata['IndexStride']['IndexC'])
        except Exception:
            pass
    if annotID:
        regionparams['output'] = {
            'maxWidth': int(ts.sizeX // reduce * rscale),
            'maxHeight': int(ts.sizeY // reduce * rscale),
        }
        img = ts.getRegion(**regionparams)[0]
        debug_image(debug, img, 'source')
        gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
        gc.token = args.girderToken
        try:
            annot = gc.get(f'annotation/{annotID.strip()}')
        except Exception as exc:
            try:
                annotID = gc.get('annotation',
                                 parameters={'name': annotID.strip(), 'limit': 1})[0]['_id']
                annot = gc.get(f'annotation/{annotID.strip()}')
            except Exception:
                raise exc
        if annot['annotation']['elements'][0]['type'] == 'point':
            return annot['annotation']['elements']
        img = (rasterio.features.rasterize(
            annotation_to_shapely(annot, reduce / rscale),
            out_shape=(sizeY, sizeX)) > 0).astype('bool')
        img = 255.0 * img
        debug_image(debug, img, 'annotation')
    else:
        regionparams['output'] = {
            'maxWidth': int(ts.sizeX // reduce * rscale),
            'maxHeight': int(ts.sizeY // reduce * rscale),
        }
        img = ts.getRegion(**regionparams)[0]
        print(f'Region shape {img.shape}')
        if len(img.shape) == 3 and img.shape[-1] >= 3:
            img = img[:, :, :3]
            debug_image(debug, img, 'source')
            # possibly get from args
            print('Deconvolving')
            stains = ['hematoxylin', 'eosin', 'null']
            stain_matrix = np.array([stain_color_map[stain] for stain in stains]).T
            # stain_matrix = np.array([[0.644211, 0.716556, 0.266844],
            #                          [0.175411, 0.972178, 0.154589],
            #                          [0, 0, 0]]).T
            img = histomicstk.preprocessing.color_deconvolution.color_deconvolution(
                img, stain_matrix).Stains[:, :, 0]
            img = 255 - img
            debug_image(debug, img, 'deconvolved')
        elif len(img.shape) == 3:
            print('Using directly')
            img = img[:, :, 0]
            debug_image(debug, img, 'source')

        if args.threshold:
            print('Thresholding')
            img = (img > skimage.filters.threshold_otsu(img))
            # img = (skimage.filters.threshold_local(
            #        img / 255, block_size=151 // reduce, offset=-0.25) > 0.5)
            debug_image(debug, img, 'threshold')
            print(f'Thresholded, non-zero pixels: {np.sum(img)}')
            if args.smallObject:
                img = skimage.morphology.remove_small_objects(img, args.smallObject / reduce)
                print(f'Removed small objects, non-zero pixels: {np.sum(img)}')
                debug_image(debug, img, 'removedSmallA')
            if args.disk:
                img = skimage.morphology.binary_opening(
                    img, skimage.morphology.disk(max(args.disk / reduce, 1)))
                print(f'Binary opening, non-zero pixels: {np.sum(img)}')
                debug_image(debug, img, 'binaryOpening')
            if args.smallObject:
                img = skimage.morphology.remove_small_objects(img, args.smallObject / reduce)
                print(f'Removed small objects, non-zero pixels: {np.sum(img)}')
                debug_image(debug, img, 'removedSmallB')
            img = 255.0 * img

        debug_image(debug, img, 'processed')
    return img


def debug_image(debug, img, tag):
    """
    An an image as a frame of a debug image with a unique channel tag.  Added
    images are always at least 3 channel.

    :param debug: a large_image sink image.  None to be a no-op.
    :param img: the image to add.  If a numpy array, this will be contrast
        extended and, if single channel, made to be 3 channels.
    :param tag: string for channel name.
    """
    if not debug:
        return
    if isinstance(img, np.ndarray):
        if len(img.shape) == 2:
            img = np.resize(img, (img.shape[0], img.shape[1], 1))
        if img.shape[-1] == 1:
            img = np.repeat(img[:, :, 0:1], 3, axis=2)
        img = img.astype(float)
        if np.amax(img) > 0:
            img *= 255.0 / np.amax(img)
        img = img.astype(np.uint8)
    basetag = tag
    basenum = 1
    while tag in (debug.channelNames or []):
        basenum += 1
        tag = f'{basetag}{basenum}'
    debug.addTile(img, c=len(debug.channelNames or []))
    channelNames = (debug.channelNames or [])
    channelNames.append(tag)
    debug.channelNames = channelNames


def output_image(tmpdir, name, multi, outpath):
    if outpath.rsplit('.')[-1].lower() in {'yaml', 'yml'}:
        multi = copy.deepcopy(multi)
        for source in multi['sources']:
            source['path'] = os.path.join('.', source['path'].rsplit(os.path.sep, 1)[-1])
        open(outpath, 'w').write(yaml.dump(multi))
    else:
        yamlpath = os.path.join(tmpdir, name)
        open(yamlpath, 'w').write(yaml.dump(multi))
        large_image_converter.convert(yamlpath, outpath)


def transform_images(ts1, ts2, matrix, out2path=None, outmergepath=None, rscale=1):
    if hasattr(matrix, 'tolist'):
        matrix = matrix.tolist()
    sx, sy = ts1.sizeX, ts1.sizeY
    for cx, cy in [(0, 0), (ts2.sizeX, 0), (0, ts2.sizeY), (ts2.sizeX, ts2.sizeY)]:
        txy = np.array(matrix).dot([[cx], [cy], [1]])
        sx = int(math.ceil(max(sx, txy[0])))
        sy = int(math.ceil(max(sy, txy[1])))
    trans2 = {
        'name': f'Transform of {os.path.basename(ts2.largeImagePath)}',
        'width': sx,
        'height': sy,
        'backgroundColor': [0, 0, 0],
        'scale': {},
        'dtype': str(ts2.dtype) if np.dtype(ts2.dtype).kind != 'f' else 'uint16',
        'sources': [{
            'path': ts2.largeImagePath,
            'position': {
                's11': matrix[0][0],
                's12': matrix[0][1],
                'x': matrix[0][2],
                's21': matrix[1][0],
                's22': matrix[1][1],
                'y': matrix[1][2],
            },
        }],
    }
    if ts2.frames > 1:
        trans2['singleBand'] = True
    for k in {'mm_x', 'mm_y', 'magnfication'}:
        val = ts2.metadata.get(k) or ts1.metadata.get(k) or 0
        if val:
            trans2['scale'][k] = val / rscale
    print('---')
    print(yaml.dump(trans2, sort_keys=False))

    combo = copy.deepcopy(trans2)
    combo['name'] = (
        f'Transform of {os.path.basename(ts2.largeImagePath)} with '
        f'{os.path.basename(ts1.largeImagePath)}')
    if ts1.frames == 1 and ts2.frames == 1:
        combo['sources'].append({
            'path': ts1.largeImagePath,
            'z': 1,
        })
    elif ts1.frames == 1 and ts2.frames > 1:
        for band in range(1 if ts1.bandCount < 3 else 3):
            combo['sources'].append({
                'path': ts1.largeImagePath,
                'channel': ['red', 'green', 'blue'][band] if ts1.bandCount >= 3 else 'gray',
                'z': 0,
                'c': band + (
                    ts2.frames
                    if ts2.metadata.get('IndexRange', {}).get('IndexC', 0) <= 1 else
                    ts2.metadata['IndexRange']['IndexC']),
                'style': {'dtype': 'uint8', 'bands': [{'band': band + 1, 'palette': 'white'}]},
            })
    else:
        combo['singleBand'] = True
        if ts2.frames == 1:
            src = combo['sources'].pop()
            for band in range(1 if ts2.bandCount < 3 else 3):
                src = src.copy()
                src.update({
                    'channel': ['red', 'green', 'blue'][band] if ts2.bandCount >= 3 else 'gray',
                    'z': 0,
                    'c': band,
                    'style': {'dtype': 'uint8', 'bands': [{'band': band + 1, 'palette': 'white'}]},
                })
                combo['sources'].append(src)
            if 'channels' in ts1.metadata:
                combo['channels'] = [
                    s['channel'] for s in combo['sources']] + ts1.metadata['channels']
        combo['sources'].append({
            'path': ts1.largeImagePath,
            'z': 0,
            'c': (len(combo['sources'])
                  if ts2.metadata.get('IndexRange', {}).get('IndexC', 0) <= 1 else
                  ts2.metadata['IndexRange']['IndexC']),
        })
    print('---')
    print(yaml.dump(combo, sort_keys=False))
    print('\n---')
    with tempfile.TemporaryDirectory() as tmpdir:
        sys.stdout.flush()
        logger = logging.getLogger('large_image')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger = logging.getLogger('large-image-converter')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        if out2path:
            output_image(tmpdir, 'out2transform.yaml', trans2, out2path)
        if outmergepath:
            output_image(tmpdir, 'outmergetransform.yaml', combo, outmergepath)


def register_points(args, points1, points2):
    if not isinstance(points1, list) or not isinstance(points2, list):
        msg = 'If one annotation is points, both images must specify a point annotation'
        raise Exception(msg)
    labels = {}
    allpts = []
    for key, points in [('1', points1), ('2', points2)]:
        for idx, pt in enumerate(points):
            if len(allpts) < idx + 1:
                allpts.append({})
            allpts[idx][key] = [pt['center'][0], pt['center'][1]]
            label = pt.get('label', {}).get('value')
            if label:
                labels.setdefault(label, {})
                labels[label][key] = [pt['center'][0], pt['center'][1]]
    labels = {k: v for k, v in labels.items() if len(v) == 2}
    if len(labels) >= 2:
        allpts = list(labels.values())
        print(f'Using {len(allpts)} labeled points: {sorted(labels.keys())}')
    else:
        allpts = [pt for pt in allpts if len(pt) == 2]
        print(f'Using {len(allpts)} corresponding points')
    mat = []
    val2 = []
    for pts in allpts:
        pt1 = pts['1']
        pt2 = pts['2']
        val2.append(pt2[0])
        val2.append(pt2[1])
        if args.transform == 'AFFINE':
            mat.append([pt1[0], pt1[1], 1, 0, 0, 0])
            mat.append([0, 0, 0, pt1[0], pt1[1], 1])
        else:
            mat.append([pt1[0], pt1[1], 1, 0])
            mat.append([pt1[1], -pt1[0], 0, 1])
    proj = np.linalg.lstsq(mat, np.array([val2]).T, rcond=-1)[0].flatten().tolist()
    print(proj, mat, val2)
    if args.transform == 'AFFINE':
        return np.array([[proj[0], proj[1], proj[2]], [proj[3], proj[4], proj[5]], [0, 0, 1]])
    scale = (proj[0] ** 2 + proj[1] ** 2) ** 0.5
    proj[0] /= scale
    proj[1] /= scale
    return np.array([[proj[0], proj[1], proj[2]], [-proj[1], proj[0], proj[3]], [0, 0, 1]])


def main(args):
    print('\n>> CLI Parameters ...\n')
    pprint.pprint(vars(args))
    if len({args.image2, args.image1} & {
            args.outputDebugImage, args.outputSecondImage, args.outputMergedImage}):
        msg = ('One of the output images has the same file name as one of '
               'the input images.  Note that this may be the Girder file '
               'name rather than the item name.')
        raise Exception(msg)
    if ((args.outputDebugImage and (
            args.outputDebugImage == args.outputSecondImage or
            args.outputDebugImage == args.outputMergedImage)) or (
            args.outputSecondImage and
            args.outputSecondImage == args.outputMergedImage)):
        msg = 'Two of the output images have the same file name.'
        raise Exception(msg)
    if ((args.outputDebugImage and os.path.exists(args.outputDebugImage)) or
            (args.outputSecondImage and os.path.exists(args.outputSecondImage)) or
            (args.outputMergedImage and os.path.exists(args.outputMergedImage))):
        msg = 'One of the output images already exists.'
        raise Exception(msg)
    if not args.style1 or args.style1.startswith('{#control'):
        args.style1 = None
    if not args.style2 or args.style2.startswith('{#control'):
        args.style2 = None
    with ProgressHelper('Register Images', 'Registering images', args.progress) as prog:
        debug = large_image.new() if args.outputDebugImage else None
        prog.message('Opening first image')
        ts1 = large_image.open(args.image1, style=args.style1)
        print('Image 1:')
        pprint.pprint(ts1.metadata)
        prog.message('Opening second image')
        ts2 = large_image.open(args.image2, style=args.style2)
        print('Image 2:')
        pprint.pprint(ts2.metadata)
        rscale = 1
        if (args.internalscale and ts1.metadata['mm_x'] and
                ts1.metadata['mm_y'] and ts2.metadata['mm_x'] and ts2.metadata['mm_y']):
            rscale = math.sqrt(
                ts2.metadata['mm_x'] * ts2.metadata['mm_y'] /
                (ts1.metadata['mm_x'] * ts1.metadata['mm_y']))
            print(f'Image relative scale: {rscale:5.3f}')
        maxRes = int(max(ts1.sizeX, ts1.sizeY, ts2.sizeX * rscale, ts2.sizeY * rscale))
        reduce = 1
        while math.ceil(maxRes / reduce) >= args.maxResolution:
            reduce *= 2
        print(f'Using reduction factor of {reduce}' if reduce > 1 else
              'Using images at original size')
        sizeX = int(math.ceil(max(ts1.sizeX, ts2.sizeX * rscale, ts2.sizeY * rscale) / reduce))
        sizeY = int(math.ceil(max(ts1.sizeY, ts2.sizeX * rscale, ts2.sizeY * rscale) / reduce))
        print(f'Registration size {sizeX} x {sizeY}')

        prog.message('Fetching first image')
        img1 = get_image(ts1, sizeX, sizeY, args.frame1, args.annotationID1,
                         args, reduce, debug)
        prog.message('Fetching second image')
        img2 = get_image(ts2, sizeX, sizeY, args.frame2, args.annotationID2,
                         args, reduce, debug, rscale)
        if isinstance(img1, list) or isinstance(img2, list):
            full = register_points(args, img1, img2)
        else:
            img1 = np.pad(img1, ((0, sizeY - img1.shape[0]),
                                 (0, sizeX - img1.shape[1])), mode='constant')
            prog.message('Registering')
            sr = pystackreg.StackReg(getattr(
                pystackreg.StackReg, args.transform, pystackreg.StackReg.AFFINE))

            # check four cardinal rotations
            print('Check rotations')
            best = None
            rotMatrix = {
                0: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                1: [[0, -1, img2.shape[1]], [1, 0, 0], [0, 0, 1]],
                2: [[-1, 0, img2.shape[1]], [0, -1, img2.shape[0]], [0, 0, 1]],
                3: [[0, 1, 0], [-1, 0, img2.shape[0]], [0, 0, 1]],
            }
            for rot in range(4 if args.rotations else 1):
                rimg = np.rot90(img2, rot)
                rimg = np.pad(rimg, ((0, sizeY - rimg.shape[0]),
                                     (0, sizeX - rimg.shape[1])), mode='constant')
                timg = sr.register_transform(img1, rimg)
                debug_image(debug, timg, f'rotation{rot * 90}deg')
                print(f'Rotation matrix {rot * 90} deg')
                print(sr.get_matrix())
                if best is None or np.sum(np.abs(timg - img1)) < best[1]:
                    best = rot, np.sum(np.abs(timg - img1)), timg, sr.get_matrix().copy()
                    print(f'Current best rotation is {rot * 90} deg, score {best[1]}')
            if best[0]:
                print(f'Use rotation of {best[0] * 90} degrees ccw')

            print('Register plain')
            rot = best[0]
            timg = best[2]
            debug_image(debug, timg, 'transformed')
            prog.message('Registered')
            print('Direct result')
            full = best[3]
            print(full)
            print('With rotation')
            full = np.dot(rotMatrix[best[0]], full)
            print(full)
            if rscale != 1:
                full = np.dot([[1. / rscale, 0, 0], [0, 1. / rscale, 0], [0, 0, 1]], full)
            full[0][2] *= reduce
            full[1][2] *= reduce
        print('Full result')
        print(full)
        print('Inverse')
        inv = np.linalg.inv(full)
        print(inv)
        print('Transforming image')
        prog.message('Transforming')
        transform_images(ts1, ts2, inv, args.outputSecondImage, args.outputMergedImage, rscale)
        debug.write(args.outputDebugImage) if debug else None


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
