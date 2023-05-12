import os
import sys
from typing import IO, Union


def _get_or_download_data(filename: str, origin: str, asbytes: bool = False) -> Union[IO[str], IO[bytes]]:
    homedir = os.path.expanduser('~')
    datadir = os.path.join(homedir, '.shorttext')
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    targetfilepath = os.path.join(datadir, filename)
    if not os.path.exists(targetfilepath):
        print('Downloading...')
        print('Source: ', origin)
        print('Target: ', targetfilepath)
        try:
            urlretrieve(origin, targetfilepath)
        except:
            print('Failure to download file!')
            print(sys.exc_info())
            os.remove(targetfilepath)

    return open(targetfilepath, 'rb' if asbytes else 'r')