import argparse
from model_v2.VGG import *
from model_v2.Decoder import *
from model_v2.Transform import *
import os
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
import glob

def test_transform():
    transform_list = [transforms.Resize(size=(512, 512))]
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

parser = argparse.ArgumentParser()

# Basic options
parser.add_argument('--content', type=str, default='../content/*',
                    help='File path to the content image')
parser.add_argument('--style', type=str, default='./style/*',
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--steps', type=str, default=1)
parser.add_argument('--content_encoder', type=str, default='../NST_v2/experiments3/content_encoder_iter_600000.pth')
parser.add_argument('--style_encoder', type=str, default='../NST_v2/experiments3/style_encoder_iter_600000.pth')
parser.add_argument('--decoder', type=str, default='../NST_v2/experiments3/decoder_iter_600000.pth')
parser.add_argument('--transform', type=str, default='../NST_v2/experiments3/transformer_iter_600000.pth')

# Additional options
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='../output2',
                    help='Directory to save the output image(s)')

# Advanced options
args = parser.parse_args('')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

if not os.path.exists(args.output):
    os.mkdir(args.output)

content_encoder = VGG19()
style_encoder = VGG19()
transform = Transform(in_planes=512)
decoder = Decoder('Decoder_v2')


decoder.eval()
transform.eval()
content_encoder.eval()
style_encoder.eval()

decoder.load_state_dict(torch.load(args.decoder))
transform.load_state_dict(torch.load(args.transform))
content_encoder.load_state_dict(torch.load(args.content_encoder))
style_encoder.load_state_dict(torch.load(args.style_encoder))

transform.to(device)
decoder.to(device)
content_encoder.to(device)
style_encoder.to(device)

content_tf = test_transform()
style_tf = test_transform()

style_files = glob.glob(args.style)
content_files = glob.glob(args.content)
for content_file in content_files:
    content = content_tf(Image.open(content_file))
    content = content.to(device).unsqueeze(0)
    for style_file in style_files:
        # style = style_tf(Image.open(args.style))
        print(content_file, style_file)
        style_img = Image.open(style_file)
        print(style_img.mode)
        if style_img.mode != "RGB":
            style_img = style_img.convert('RGB')
        style = style_tf(style_img)
        style = style.to(device).unsqueeze(0)

        with torch.no_grad():
            for x in range(args.steps):
                print('iteration ' + str(x))

                content_feas = content_encoder(content)
                content_feas_s = content_encoder(style)
                style_feas = style_encoder(style)
                stylized_fea = transform(content_feas['conv4_1'], content_feas_s['conv4_1'], style_feas['conv4_1'],
                                              content_feas['conv5_1'], content_feas_s['conv5_1'], style_feas['conv5_1'])
                stylized_img = decoder(stylized_fea)

                stylized_img.clamp(0, 255)

            stylized_img = stylized_img.cpu()

            # output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
            #     args.output, splitext(basename(args.content))[0],
            #     splitext(basename(args.style))[0], args.save_ext
            # )
            output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
                    args.output, splitext(basename(content_file))[0],
                    splitext(basename(style_file))[0], args.save_ext
                )
            save_image(stylized_img, output_name)