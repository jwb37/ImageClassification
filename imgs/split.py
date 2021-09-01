import os
import sys
import random
import numpy as np
from pathlib import Path

ImageSuffixes = set([
    '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'
])

def split_set(base_dir, train_split, test_split, valid_split):
    ratios = np.array([train_split, test_split, valid_split], dtype=float)
    ratios /= np.sum(ratios)

    paths = dict()
    paths['base'] = Path(base_dir)
    paths['source'] = paths['base'] / 'all'
    for phase in ['train', 'test', 'valid']:
        paths[phase] = paths['base'] / phase
        if not paths[phase].exists():
            paths[phase].mkdir()

    for subdir in [path for path in paths['source'].iterdir() if path.is_dir()]:
        category_imgs = set([path for path in subdir.iterdir() if path.suffix in ImageSuffixes])
        size = dict()
        # Calculate number of image files in each split
        size['train'], size['test'], size['valid'] = (ratios * len(category_imgs)).astype(int)
        # Total of splits should equal total number of images.
        # If rounding causes otherwise, adjust 'train' set to fit.
        size['train'] += len(category_imgs) - sum(size.values())

        print( f"Category: {subdir.name}, Total pictures: {len(category_imgs)}" )
        print( f"\tSplits: Train({size['train']}), Test({size['test']}), Validation({size['valid']})" )

        for phase in ['train', 'test', 'valid']:
            sample = random.sample(list(category_imgs), size[phase])

            for file_path in sample:
                target_path = paths[phase] / subdir.name
                if not target_path.exists():
                    target_path.mkdir()
                target_path /= file_path.name
                rel_file_path = Path( '..', '..', *file_path.parts[1:] )
                target_path.symlink_to(rel_file_path)

            category_imgs -= set(sample)

if __name__ == '__main__':
    if( len(sys.argv) != 2 ):
        print( "Run this program with a single argument (the target subdir containing your dataset)" )
        exit()

    split_set(sys.argv[1], 90, 5, 5)
