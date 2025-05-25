import os
import clip
import torch


# get the text-based organ/tumor names embeddings

ORGAN_NAME = ['Spleen', 'Right Kidney', 'Left Kidney', 'Gall Bladder', 'Esophagus', 
                'Liver', 'Stomach', 'Arota', 'Postcava', 'Portal Vein and Splenic Vein',
                'Pancreas', 'Right Adrenal Gland', 'Left Adrenal Gland', 'Duodenum', 'Hepatic Vessel',
                'Right Lung', 'Left Lung', 'Colon', 'Intestine', 'Rectum', 
                'Bladder', 'Prostate', 'Left Head of Femur', 'Right Head of Femur', 'Celiac Truck',
                'Kidney Tumor', 'Liver Tumor', 'Pancreas Tumor', 'Hepatic Vessel Tumor', 'Lung Tumor', 
                'Colon Tumor', 'Kidney Cyst']

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)


text_inputs = torch.cat([clip.tokenize(f'A computerized tomography of a {item}') for item in ORGAN_NAME]).to(device)

# Calculate text embedding features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    print(text_features.shape, text_features.dtype) 
    torch.save(text_features, 'txt_encoding.pth')



# get the text-based task embeddings

TASK_NAME = ['Spleen, Right Kidney, Left Kidney, Gall Bladder, Esophagus, Liver, Stomach, Aorta, Postcava, Portal Vein, Splenic Vein, Pancreas, Right Adrenal Gland, Left Adrenal Gland, Duodenum',
             'Spleen, Left Kidney, Gall Bladder, Esophagus, Liver, Stomach, Pancreas, Duodenum',
             'Liver',
             'Liver, Liver Tumor',
             'Right Kidney, Left Kidney, Kidney Tumor, Kidney Cyst',
             'Liver, Spleen, Left Kidney, Right Kidney, Stomach, Gall Bladder, Esophagus, Pancreas, Duodenum, Colon, Intestine, Right Adrenal Gland, Left Adrenal Gland, Rectum, Bladder, Left Head of Femur, Right Head of Femur',
             'Liver, Right Kidney, Left Kidney, Spleen, Pancreas',
             'Spleen, Right Kidney, Left Kidney, Gall Bladder, Esophagus, Liver, Stomach, Aorta, Postcava, Pancreas, Right Adrenal Gland, Left Adrenal Gland, Duodenum, Bladder, Prostate',
             'Liver, Bladder, Right Lung, Left Lung, Right Kidney, Left Kidney',
             'Liver, Right Kidney, Spleen, Pancreas, Aorta, Postcava, Stomach, Gall Bladder, Esophagus, Right Adrenal Gland, Left Adrenal Gland, Celiac Truck',
             'Spleen, Liver, Pancreas, Liver Tumor, Lung Tumor, Pancreas Tumor, Hepatic Vessel, Hepatic Vessel Tumor, Colon Tumor']
             

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
text_inputs = torch.cat([clip.tokenize(f'A task of segmenting {item} in computerized tomography') for item in TASK_NAME]).to(device)

# Calculate text embedding features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    print(text_features.shape, text_features.dtype) 
    torch.save(text_features, 'txt_task_encoding.pth')

