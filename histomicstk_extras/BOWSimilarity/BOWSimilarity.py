import json
import pprint

import girder_client
from slicer_cli_web import CLIArgumentParser


def main(args):  # noqa
    print('\n>> CLI Parameters ...\n')
    pprint.pprint(vars(args))

    isJsonList = open(args.wordfile).read(1)[:1] == '{'

    features = {}
    radius = 0
    if isJsonList:
        for line in open(args.wordfile).readlines():
            try:
                line = json.loads(line)
            except Exception:
                continue
            x = int(line['params']['tile']['gx'] + line['params']['tile']['gwidth'] / 2)
            y = int(line['params']['tile']['gy'] + line['params']['tile']['gheight'] / 2)
            words = line['result']
            words = {w.strip(',.-').lower() for w in words.split()}
            features[(x, y)] = words
    else:
        wordset = open(args.wordfile).read()
        prefix = wordset.split(':', 1)[0].split(',', 1)[0]
        lines = ('\n' + wordset).split('\n' + prefix)[1:]
        for line in lines:
            parts = line.split(':', 1)[0].split(',')
            x, y = int(parts[1]), int(parts[2])
            w, h = int(parts[3]), int(parts[4])
            x += w // 2
            y += h // 2
            radius = max(radius, w // 2, h // 2)
            words = line.split(':', 1)[1].strip()
            # set of words
            words = {w.strip(',.-').lower() for w in words.split()}
            features[(x, y)] = words
    allcommon = None
    for words in features.values():
        if allcommon is None:
            allcommon = words.copy()
        else:
            allcommon &= words
    print(f'Common words: {" ".join(sorted(allcommon))}')
    for words in features.values():
        words -= allcommon
    minx = min(x for x, y in features)
    miny = min(y for x, y in features)
    spacex = min(x for x, y in features if x != minx) - minx
    try:
        spacey = min(y for x, y in features if y != miny) - miny
    except Exception:
        spacey = spacex
    tx, ty = list(features)[0]
    for x, y in features:
        if (((args.keypoint[0] - x) ** 2 + (args.keypoint[1] - y) ** 2) <
                ((args.keypoint[0] - tx) ** 2 + (args.keypoint[1] - ty) ** 2)):
            tx, ty = x, y
    print(f'Using {tx}, {ty} as focus (asked for {args.keypoint[0]}, {args.keypoint[1]})')
    wk = features[(tx, ty)]
    print(f'Words associated with focus: {" ".join(sorted(wk))}')

    heatmap = {
        'type': 'heatmap',
        'points': [],
        'radius': max(spacex, spacey) * 2.5,
        'rangeValues': [0, 1],
        'scaleWithZoom': True,
    }
    for (x, y) in sorted(features):
        wf = features[(x, y)]
        diff = len(wf - wk) + len(wk - wf)
        val = max(0, (len(wk) - diff) / len(wk))
        if val > 0:
            heatmap['points'].append([x, y, 0, val])
    annot = {
        'name': f'Heatmap {tx}, {ty}',
        'elements': [heatmap],
    }
    print(json.dumps(heatmap))
    print(f'{len(heatmap["points"])} total points')
    if args.girderApiUrl:
        gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
        gc.token = args.girderToken
        if args.annotationID:
            gc.put(f'annotation/{args.annotationID.strip()}', data=json.dumps(annot))
        else:
            try:
                itemId = gc.get(f'file/{args.image}')['itemId']
            except Exception:
                itemId = gc.get(f'item/{args.image}')['_id']
            gc.post(f'annotation/item/{itemId}', data=json.dumps(annot))


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
