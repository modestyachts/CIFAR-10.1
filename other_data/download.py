import argparse
from pathlib import Path
import tarfile
import urllib

import boto3
import botocore

parser=argparse.ArgumentParser(description='Download s3 files.')
parser.add_argument('--all', action='store_true',
                    help='Download all files')
parser.add_argument('filename', default=None, nargs='?', type=str, 
                    help='Filename to download')
parser.add_argument('--force', action='store_true',
                    help='Download a file even if it already exists locally')
args = parser.parse_args()

s3 = boto3.resource('s3')

BUCKET_NAME = 'cifar-10-1'

ALL_FILENAMES = [
  'tinyimage_large_dst_images_v6.1.json',
  'tinyimage_large_dst_image_data_v6.1.pickle',
  'tinyimage_large_dst_images_v4.json',
  'tinyimage_large_dst_image_data_v4.pickle',
  'cifar10_keywords.json',
  'cifar10_keywords_unique.json',
  'tinyimage_cifar10_distances_full.json',
]

def download_file(key, force_download):
  if Path(key).is_file() and not force_download:
    print('File {} already exists locally.'.format(key))
  else:
    try:
       print('Downloading {} ...'.format(key))
       s3.Bucket(BUCKET_NAME).download_file(key, key)
    except botocore.exceptions.ClientError as e:
      if e.response['Error']['Code'] == "404":
        print("The file {} does not exist in the s3 bucket.".format(key))
      else:
        raise

def download_cifar10(force_download):
    filename = 'cifar-10-python.tar.gz'
    dirname = 'cifar10'
    if Path(filename).is_file() and not force_download:
        print('{} already exists, not downloading.'.format(filename))
    else:
        urllib.request.urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', filename)
    if Path(dirname).is_dir() and not force_download:
        print('{} folder already exists, not extracting again.'.format(dirname))
    else:
        tf = tarfile.open(filename, 'r:gz')
        tf.extractall('.')
        tf.close()
        if Path(dirname).is_dir() and force_download:
            shutil.rmtree(cifar10)
        Path('cifar-10-batches-py').rename(dirname)


if args.all:
  for key in ALL_FILENAMES:
    download_file(key, args.force)

if args.filename:
    if args.filename == 'cifar10':
        download_cifar10()
    else:
        download_file(args.filename, args.force)
