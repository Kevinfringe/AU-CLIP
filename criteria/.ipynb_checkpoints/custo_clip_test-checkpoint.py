import torch

# # Example AU targets (dummy data)
# batch_size = 2
# num_classes = 8
# au_targets = torch.tensor([[0.3, 0.6, 0.8, 0.2, 0.7, 0.4, 0.9, 0.1],
#                            [0.9, 0.5, 0.1, 0.8, 0.6, 0.2, 0.7, 0.4]])

# # Define the AU dictionary and template
# au_dict = {
#     0: "inner eyebrow raise",
#     1: "outter eyebrow raise",
#     2: "brow lower",
#     3: "cheek raise",
#     4: "nose wrinkle",
#     5: "lip corner raise up",
#     6: "mouth open",
#     7: "jaw drop"
# }

# au_template = "human face with "

# # Process the AU targets
# for i in range(batch_size):
#     au_entry = au_targets[i]
#     chosen_au_indices = torch.nonzero(au_entry > 0.5).squeeze(1)
#     topk_indices = torch.topk(au_entry[chosen_au_indices], k=min(3, len(chosen_au_indices)), dim=0).indices
#     chosen_au_indices = chosen_au_indices[topk_indices]
#     chosen_au = [au_dict[index.item()] for index in chosen_au_indices]
#     additional_description = au_template + ', '.join(chosen_au)
    
#     print("AU targets for image", i + 1)
#     print("AU entry:", au_entry)
#     print("Chosen AU indices:", chosen_au_indices)
#     print("Chosen AU descriptions:", chosen_au)
#     print("Additional description:", additional_description)
#     print("-" * 30)

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


# Example result and additional_description strings
result = "happy"
additional_description = "human face with outter eyebrow raise, lip corner raise up, mouth open"

# Combine result and additional_description separately into common_template
combined_template = []
for template in common_template:
    template_result = template.format(result)
    template_additional = template.format(additional_description)
    combined_template.extend([template_result, template_additional])

# Print the combined templates
for template in combined_template:
    print(template)

print('-'*30)
print(len(common_template))
print(len(combined_template))