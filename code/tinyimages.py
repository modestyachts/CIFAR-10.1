# Derived from https://github.com/cioc/PyTinyImage/blob/master/tinyimage.py

import os

import numpy
import tqdm
import scipy


def strcmp(str1, str2):
  l = min(len(str1), len(str2)) 
  for i in range(0, l):
    if ord(str1[i]) > ord(str2[i]):
      return 1
    if ord(str1[i]) < ord(str2[i]):
      return -1
  if len(str1) > len(str2):
    return 1
  if len(str1) < len(str2):
    return -1
  return 0


class TinyImages(object):
    def __init__(self, data_path):
        meta_file_path = os.path.join(data_path, 'tiny_metadata.bin')
        data_file_path = os.path.join(data_path, 'tiny_images.bin')
        self.meta_file = open(meta_file_path, "rb")
        self.data_file = open(data_file_path, "rb")
        self.img_count = 79302017

    def get_metadata(self, indx):
        # Only keyword and filename are correctly decoded at the moment.
        # This is OK because we only use keyword in our scripts.
        offset = indx * 768
        self.meta_file.seek(offset)
        data = self.meta_file.read(768)
        keyword = str(data[0:80], 'utf-8').strip()
        filename = str(data[80:185], 'utf-8').split(' ')[0]
        width = data[185:187]
        height = data[187:189]
        color = data[189:190]
        date = data[190:222]
        engine = data[222:232]
        thumbnail = data[232:432]
        source = data[432:760]
        page = data[760:764]
        indpage = data[764:768]
        indengine = data[768:762]
        indoverall = data[762:764]
        label = data[764:768]
        return (keyword, filename, width, height, color, date, engine, thumbnail,
                source, page, indpage, indengine,indoverall, label)

    def get_keywords(self, show_progress=False):
      result = []
      vals = range(self.img_count)
      if show_progress:
          vals = tqdm.tqdm(vals)
      for ii in vals:
          result.append(self.get_metadata(ii)[0])
      return result

    def binary_search(self, term):
        # Comment added by Ludwig: the binary search here is not a "full"
        # binary search but only 10 iterations. This is OK because it is
        # only used by retrieve_by_term below, and that function is robust
        # to start and end points that are too low / too high (i.e.,
        # conservative in the sense that they include the range of interest
        # and more).
        low = 0
        high = self.img_count
        for i in range(0, 9):
            meta = self.get_metadata(int((low + high) / 2))
            cmp = strcmp(meta[0].lower(), term.lower())
            if (cmp == 0):
                return (low, high)
            if (cmp == 1):
                high = (low + high) // 2
            if (cmp == -1):
                low = (low + high) // 2
        return (low, high)

    def retrieve_by_term(self, search_term, max_pics):
        (l, h) = self.binary_search(search_term)
        found = False
        found_count = 0
        result = []
        for i in range(l, h):
            meta = self.get_metadata(i)
            if meta[0].lower() == search_term.lower():
                found = True
                result.append(i)
                found_count += 1
                if found_count == max_pics:
                    break
            else:
                if found:
                    break  
        return result

    def slice_to_numpy(self, indx, num_images=1, reshape=True):
        offset = indx * 3072
        self.data_file.seek(offset)
        data = self.data_file.read(3072 * num_images) 
        result = numpy.fromstring(data, dtype='uint8')
        if reshape:
            if num_images == 1:
                result = result.reshape(32, 32, 3, order='F')
            else:
                result = result.reshape(32, 32, 3, num_images, order='F')
                result = result.transpose([3, 0, 1, 2])
        return result

    def close(self):
        self.data_file.close()
       	self.meta_file.close()
