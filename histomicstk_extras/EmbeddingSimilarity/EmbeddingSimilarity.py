import json
import os
import pprint
import sys
import time

import girder_client
import numpy as np
import tqdm
from slicer_cli_web import CLIArgumentParser


class GigapathModel:
    patch = 224
    magnification = 20
    model_name = 'hf_hub:prov-gigapath/prov-gigapath'

    def prepare(self):
        import timm
        import torch
        from torchvision import transforms

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tile_encoder = timm.create_model(self.model_name, pretrained=True)
        self.tile_encoder = self.tile_encoder.to(self.device)
        self.tile_encoder.eval()  # skip

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def infer(self, imgs):
        import torch

        imgs = [np.copy(img) if not img.flags['WRITEABLE'] else img for img in imgs]
        imgs = torch.stack([self.transformer(img[:, :, :3]) for img in imgs], axis=0)
        imgs = imgs.to(self.device)
        return self.tile_encoder(imgs).to('cpu').numpy()


class DivoV2Model:
    patch = 224
    magnification = 20
    model_name = 'facebook/dinov2-large'

    def prepare(self):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = AutoImageProcessor.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

    def infer(self, imgs):
        import torch

        imgs = torch.stack([torch.from_numpy(img[:, :, :3]) for img in imgs], axis=0)
        inputs = self.processor(images=imgs, return_tensors='pt').to(self.device)
        results = self.model(**inputs)
        results = torch.mean(results.last_hidden_state, dim=1)
        return results.to('cpu').numpy()


class MidnightModel:
    patch = 224
    magnification = 20
    model_name = 'kaiko-ai/midnight'

    def prepare(self):
        import torch
        from torchvision import transforms
        from transformers import AutoModel

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def infer(self, imgs):
        import torch

        imgs = [np.copy(img) if not img.flags['WRITEABLE'] else img for img in imgs]
        imgs = torch.stack([self.transformer(img[:, :, :3]) for img in imgs], axis=0)
        imgs = imgs.to(self.device)
        results = self.model(imgs).last_hidden_state
        cls_embedding, patch_embeddings = results[:, 0, :], results[:, 1:, :]
        results = torch.cat([cls_embedding, patch_embeddings.mean(1)], dim=-1)
        return results.to('cpu').numpy()


ModelList = {
    'Gigapath': GigapathModel,
    'DinoV2Large': DivoV2Model,
    'Midnight': MidnightModel,
}


def collect_batch(model, batch, batchcoor, out, maxx=0, maxy=0):
    if len(batch) == 0:
        return out
    embeds = model.infer(batch)
    for idx in range(len(batch)):
        x, y = batchcoor[idx]
        embed = embeds[idx]
        if out is None:
            out = np.zeros((maxy, maxx, embed.shape[0]), dtype=embed.dtype)
        out[y, x, :] += embed
    return out


def generate_embedding(args):
    import large_image
    import torch

    model = ModelList[args.model]()
    print(f'Preparing model {model.model_name}')
    model.prepare()
    print('Model prepared')
    ts = large_image.open(args.image)
    if args.tilesize < model.patch:
        args.tilesize = model.patch
    if args.stride > args.tilesize or args.stride < 1:
        args.stride = args.tilesize
    numsub = (args.tilesize + model.patch - 1) // model.patch
    substep = model.patch - ((model.patch * numsub - args.tilesize) // (
        numsub - 1)) if numsub > 1 else model.patch
    out = None
    batch = []
    batchcoor = []
    if args.batch < 1:
        args.batch = 1
    mag = model.magnification if args.magnification <= 0 else args.magnification
    scale = (ts.metadata['magnification'] or 20) / mag
    with torch.no_grad():
        for tile in tqdm.tqdm(ts.tileIterator(
            scale={'magnification': mag},
            tile_size={'width': args.tilesize, 'height': args.tilesize},
            tile_overlap={'x': args.tilesize - args.stride, 'y': args.tilesize - args.stride},
            format=large_image.constants.TILE_FORMAT_NUMPY,
        ), mininterval=1 if os.isatty(sys.stdout.fileno()) else 30):
            if tile['width'] < args.tilesize or tile['height'] < args.tilesize:
                continue
            for dj in range(numsub):
                for di in range(numsub):
                    subimg = tile['tile'][
                        dj * substep: dj * substep + model.patch,
                        di * substep: di * substep + model.patch, :]
                    batch.append(subimg)
                    batchcoor.append((tile['level_x'], tile['level_y']))
                    if len(batch) == args.batch or out is None:
                        out = collect_batch(
                            model, batch, batchcoor, out,
                            tile['iterator_range']['level_x_max'],
                            tile['iterator_range']['level_y_max'])
                        batch = []
                        batchcoor = []
        out = collect_batch(model, batch, batchcoor, out)
    out /= numsub ** 2
    results = {
        'tilesize': args.tilesize * scale,
        'stride': args.stride * scale,
        'model': args.model,
        'data': out,
    }
    if args.embedout:
        np.savez(args.embedout, **results)
    return results


def main(args):  # noqa
    start = time.time()
    print('\n>> CLI Parameters ...\n')
    pprint.pprint(vars(args))

    if args.image and not args.embedin:
        embeds = generate_embedding(args)
    else:
        loaded = np.load(args.embedin, allow_pickle=True)
        embeds = {k: loaded[k].item() if loaded[k].ndim == 0 else loaded[k] for k in loaded.files}
    print(f'Data shape {embeds["data"].shape}')
    print(f'Loaded: {time.time() - start:5.3f}s elapsed')
    heatmap = {
        'type': 'heatmap',
        'points': [],
        'radius': embeds['stride'] * 2.5,
        'rangeValues': [0, 1],
        'scaleWithZoom': True,
        'colorRange': ['rgba(0, 0, 0, 0)', args.color.strip() or '#FFFF00'],
    }
    agg = max(0, args.aggregate - 1)
    tx = ty = 0
    ti = tj = 0
    for j in range(embeds['data'].shape[0] - agg):
        y = int((j + agg * 0.5) * embeds['stride'] + embeds['tilesize'] // 2)
        for i in range(embeds['data'].shape[1] - agg):
            x = int((i + agg * 0.5) * embeds['stride'] + embeds['tilesize'] // 2)
            if (((args.keypoint[0] - x) ** 2 + (args.keypoint[1] - y) ** 2) <
                    ((args.keypoint[0] - tx) ** 2 + (args.keypoint[1] - ty) ** 2)):
                tx, ty = x, y
                ti, tj = i, j
    print(f'Using {tx}, {ty} as focus (asked for {args.keypoint[0]}, {args.keypoint[1]})')
    print(f'Using {ti}, {tj} as comparison column, row')
    data = embeds['data'].astype(float)
    if agg:
        data = np.lib.stride_tricks.sliding_window_view(
            data, window_shape=(agg + 1, agg + 1, 1))[..., 0].mean(axis=(3, 4))
    norms = np.linalg.norm(data, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    data = data / norms
    match = data[tj][ti]
    print(f'Normalized: {time.time() - start:5.3f}s elapsed')
    print(match)
    similarity = np.sum(data * match, axis=-1)
    print(similarity[tj][ti])
    similarity = np.clip((similarity - args.threshold) / (1 - args.threshold), 0, 1)
    print(similarity[tj][ti])
    print(f'Similarity: {time.time() - start:5.3f}s elapsed')
    for j in range(similarity.shape[0]):
        y = int((j + agg * 0.5) * embeds['stride'] + embeds['tilesize'] // 2)
        for i in range(similarity.shape[1]):
            x = int((i + agg * 0.5) * embeds['stride'] + embeds['tilesize'] // 2)
            val = similarity[j][i]
            if i == ti and j == tj:
                print(f'Heatmap value at comparison {x}, {y}: {val}')
            if val > 0:
                heatmap['points'].append([x, y, 0, float(val)])
    annot = {
        'name': f'Heatmap {tx}, {ty}',
        'elements': [heatmap],
    }
    print(f'Heatmap: {time.time() - start:5.3f}s elapsed')
    print(f'{len(heatmap["points"])} total points')
    if args.girderApiUrl:
        gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
        gc.token = args.girderToken
        if args.annotationID:
            gc.put(f'annotation/{args.annotationID.strip()}', data=json.dumps(annot))
        else:
            try:
                itemId = gc.get(f'file/{args.imageid}')['itemId']
            except Exception:
                itemId = gc.get(f'item/{args.imageid}')['_id']
            gc.post(f'annotation/item/{itemId}', data=json.dumps(annot))
    else:
        print(json.dumps(heatmap))
    print(f'Done: {time.time() - start:5.3f}s elapsed')
    if args.imageid and not args.image or args.embedin and args.embedin == args.embedout:
        # If we are in an isolated girder job, don't output the input file
        os.unlink(args.embedout)


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
