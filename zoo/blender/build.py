"""
This downloads a finetuned Blender 90M model on EmpatheticDialogues.
"""

import os
import parlai.core.build_data as build_data
from parlai.utils.logging import logging

RESOURCES = [
    build_data.DownloadableFile(
        '1WwSy0D1KzhhOOpXmBRQMJzp0aJ2K0bDv',
        'finetuned_blender.tar.gz',
        '9eaa85dd6c7d2b7f6eb0fd70ff7e2977730ebe0252dd1740029664424311c45a',
        zipped=True, from_google=True,
    ),
]

def download(datapath, version='v1.0'):
    dpath = os.path.join(datapath, 'models', 'finetuned_blender90m')

    if not build_data.built(dpath, version):
        print('[Downloading and building Blender 90M: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        logging.info('NOTE: The download will take about 2 minutes (likely to vary depending on your internet speed)')
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version)

    return dpath
