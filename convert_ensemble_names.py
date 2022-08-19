import glob
import os
import os.path
import re


if __name__ == '__main__':

    image_paths = glob.glob('ensemble-dataset/da-vinci/*/*.tiff')
    image_regex = re.compile(r'^ensemble_(?P<id>\d+)$')

    for image_path in image_paths:
        route, basename = os.path.split(image_path)
        filename, ext = os.path.splitext(basename)

        name_match = image_regex.match(filename)

        if name_match is None:
            raise Exception(f"Couldn't match {filename}.")
        
        image_id = int(name_match.group('id'))
        new_path = os.path.join(route, f'{image_id:06}{ext}')
        print(image_path, new_path)
        os.rename(image_path, new_path)