import boto3
import botocore
import argparse
from pathlib import Path

parser=argparse.ArgumentParser(description='Download s3 files.')
parser.add_argument('--all', action='store_true',
                    help='download all files')
parser.add_argument('filename', default=None, nargs='?', type=str, 
                    help='filename to download')
args = parser.parse_args()

s3 = boto3.resource('s3')

BUCKET_NAME = 'cifar-10-1'

ALL_FILENAMES = [
  'tinyimage_large_dst_images_v4.json',
  'tinyimage_large_dst_image_data_v4.pickle',
  'tinyimage_good_indices_subselected_v4.json',
  'cifar10_keywords.json',
  'tinyimage_cifar10_distances_full.json',
]

def download_file(key):
  if Path(key).is_file():
    print('File {} already exists locally.'.format(key))
  else:
    try:
       print('Donwloading {} ...'.format(key))
       s3.Bucket(BUCKET_NAME).download_file(key, key)
    except botocore.exceptions.ClientError as e:
      if e.response['Error']['Code'] == "404":
        print("The file {} does not exist in the s3 bucket.".format(key))
      else:
        raise

if args.all:
  for key in ALL_FILENAMES:
    download_file(key)

if args.filename:
    download_file(args.filename)
