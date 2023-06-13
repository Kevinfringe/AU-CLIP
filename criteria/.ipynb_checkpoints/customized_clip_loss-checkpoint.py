'''
    A customized version of clip loss.
'''
import torch
import clip
import numpy as np
import cv2
from rmn import RMN

common_template = [
    'a bad photo of a {}.',
#    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
    ]

exp_template = "{} human face picture."

au_template = "human face with "

au_dict = {
    0 : "inner brow raise",
    1 : "outter brow raise",
    2 : "brow lower",
    3 : "cheek raise",
    4 : "nose wrinkle",
    5 : "lip corner raise up",
    6 : "mouth open",
    7 : "jaw drop"
}

class CustomizedCLIPLoss(torch.nn.Module):

    def __init__(self, opts):
        super(CustomizedCLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)
        self.fer_model = RMN()

    def forward(self, image, image_tar, au_tar):
        '''
            Unlike original version of clip loss, in this loss, the text description will be
            generated according to a SOTA FER (Facial Emotion Recognition) system, and fill the output
            class into the pre-set sentences.
        '''
        # Traverse each image within a batch.
        similarity = 0.0
        for i in range(image_tar.size(0)):
            ori_image = image[i].unsqueeze(0)
            tar_image = image_tar[i].unsqueeze(0)
            image_tensor = image_tar[i].detach().cpu()  # Get the i-th image tensor
            # Permute dimensions and convert to numpy array
            image_np = image_tensor.permute(1, 2, 0).numpy()  
            image_np = np.clip(image_np, 0, 1)  # Clip values between 0 and 1 (if necessary)

            # Convert the image to OpenCV format (BGR)
            image_cv = (image_np * 255).astype(np.uint8)
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
            
            # Predict the expression class.
            result = self.fer_model.detect_emotion_for_single_face_image(image_cv)[0]
            
            # Put new code
            au_entry = au_tar[i]
            chosen_au_indices = torch.nonzero(au_entry > 0.5)
            topk_indices = torch.topk(au_entry[chosen_au_indices], k=min(3, len(chosen_au_indices)), dim=0).indices
            chosen_au_indices = chosen_au_indices[topk_indices]
            chosen_au = [au_dict[index.item()] for index in chosen_au_indices]
            additional_description = au_template + ', '.join(chosen_au)
            if "mouth open" not in chosen_au:
                additional_description = additional_description + ', ' + 'mouth closed, lips closed.'
            additional_description += '.'
            
            # Merge two description into common_template.
            # Uncomment below block of code if you want to use the enhanced clip loss.
            # combined_template = []
            # for template in common_template:
            #     template_result = template.format(result)
            #     template_additional = template.format(additional_description)
            #     combined_template.extend([template_result, template_additional])
            
            # Comment below code if you want to use the enhanced clip loss.
            descripition = [exp_template.format(result), additional_description]
            
            # Get the tokenized description for clip.
            print("The input exp description is: " + exp_template.format(result))
            print("The input au description is: " + additional_description)
            text_inputs = torch.cat([clip.tokenize(descripition)]).cuda()
            print("Expression class for " + str(i) +"th image is: " + str(result))

        
            ori_image = self.avg_pool(self.upsample(ori_image))
            similarity += 1 - self.model(ori_image, text_inputs)[0] / 100
            
        # print(similarity.mean().shape)
        return similarity.mean() / image.size(0)