import logging
import os
import sys

from histomicstk.cli.utils import CLIArgumentParser


def main(args):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import girder_client

    import utils.demo_set as demo_set

    logger = logging.getLogger('utils.demo_set')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.debug('Parsed arguments: %r', args)

    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.token = args.girderToken
    path = None
    if args.path:
        path = gc.get(f'resource/{args.path}/path', parameters={'type': 'folder'})
    demo_set.put_demo_set(gc, args.demo, path)


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
