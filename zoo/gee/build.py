"""
This downloads a pretrained language model GEE.
It relies on the pytorch implementation of BART.
"""

import os
import parlai.core.build_data as build_data
from parlai.utils.logging import logging

RESOURCES = [
    build_data.DownloadableFile(
        'https://drive.google.com/uc?id=1TEKp3YRowAZju4UPXOSufqzU6j6_Z4wy&export=download&confirm=t',
        'gee_v1.tar.gz',
        'af0584e3c376dd364af0bbc122fea7303b9af52556f4dcbb0fbe8d6a136c0b2a',
        zipped=True, from_google=False,
    ),
]

def download(datapath, version='v1.0'):
    dpath = os.path.join(datapath, 'models', 'gee')

    if not build_data.built(dpath, version):
        print('[Downloading and building pretrained GEE: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        print('NOTE: The download can take about 8 minutes (likely to vary depending on your internet speed)')
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version)

    return dpath
