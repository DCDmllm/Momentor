from transformers import CLIPProcessor, CLIPModel
import torch

def load_clip(text_device, image_device, torch_dtype):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch_dtype)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch_dtype)
    
    clip_model.text_model = clip_model.text_model.to(text_device)
    clip_model.text_projection = clip_model.text_projection.to(text_device)
    clip_model.vision_model = clip_model.vision_model.to(image_device)
    clip_model.visual_projection = clip_model.visual_projection.to(image_device)
    
    clip_model.eval()
    
    @torch.no_grad()
    def clip_encode_image(images):
        inputs = clip_processor(images=images, return_tensors="pt").to(image_device)
        return clip_model.get_image_features(**inputs).cpu()
    
    @torch.no_grad()
    def clip_encode_text(texts):
        inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(text_device)
        return clip_model.get_text_features(**inputs).cpu()
    
    @torch.no_grad()
    def cal_clip_similarity(frames, texts):
        if isinstance(texts, str):
            texts = [texts]
        if frames.shape[0] > 1 and len(texts) == 1:
            texts = texts * frames.shape[0]
        elif frames.shape[0] == 1 and len(texts) > 1:
            frames = torch.cat([frames]*len(texts), 0)
        elif frames.shape[0] > 1 and len(texts) > 1 and frames.shape[0] != len(texts):
            assert False

        frame_features = clip_encode_image(frames)
        text_features = clip_encode_text(texts)
        return torch.cosine_similarity(frame_features, text_features)
    
    return clip_model, clip_processor, clip_encode_image, clip_encode_text