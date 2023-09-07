import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import onnxruntime


if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-folder', type=str, required=True)
    parser.add_argument('--style-folder', type=str, required=True)
    args = parser.parse_args()

    # check cmd arguments
    if not os.path.exists(args.result_folder):
        print('Cannot find the result folder: {0}'.format(args.result_folder))
        exit()
    if not os.path.exists(args.style_folder):
        print('Cannot find the style folder: {0}'.format(args.style_folder))
        exit()

    input_files = os.listdir(args.result_folder)
    style_files = os.listdir(args.style_folder)
    assert len(input_files) == len(style_files)

    # load the StyleSimiliaryDiscriminator model
    onnx_path = './StyleSimiliaryDiscriminator.onnx'
    onnx_sess = onnxruntime.InferenceSession(onnx_path)

    print('--------------------------------------------------------------------------------')
    print('result folder: {0}'.format(args.result_folder))
    print('style folder: {0}'.format(args.style_folder))
    print('--------------------------------------------------------------------------------')

    # calculate style similiary score
    style_similiary_scores = []
    pbar = tqdm(input_files, total=len(input_files), unit='file')
    for idx, fname in enumerate(pbar):

        # load input image and style image
        result_path = os.path.join(args.result_folder, fname)
        style_path = os.path.join(args.style_folder, fname)

        result_image = np.asarray(Image.open(result_path).convert('RGB').resize((512, 512))).astype(np.float32)
        style_image = np.asarray(Image.open(style_path).convert('RGB').resize((512, 512))).astype(np.float32)

        # run onnx model
        score = onnx_sess.run(['score'], {'ref': style_image, 'img': result_image})[0]

        # save score and update pbar
        style_similiary_scores.append(score)
        pbar.set_description('Avg Style Similiary Score: {0:.4f}'.format(np.mean(np.asarray(style_similiary_scores))))

print('--------------------------------------------------------------------------------')
print('Final Avg Style Similiary Score: {0:.4f}'.format(np.mean(np.asarray(style_similiary_scores))))
