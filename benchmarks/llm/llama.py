import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
import pandas as pd
import cv2
from PIL import Image
import os
from tqdm.auto import tqdm
import json
from prompts import ASK_FOR_DIAGNOSIS_PROMPT, PROMPT, SUMMARY_PROMPT, ASK_FOR_SUMMARY_PROMPT

# Configuration
MODEL_PATH = os.getenv('LLAMA_MODEL_PATH', 'meta-llama/Llama-3.2-11B-Vision-Instruct')
# PROJECT_DIR = "drive/MyDrive/EyeballProject"
SPLIT_DESC_CSV_PATH = '../input/balanced_split_desc.csv'
SPLIT_DESC_CSV_SAVE_PATH = '../output/llama_prediction.csv'

# Initialize model and processor
print(f"Loading Llama 3.2 Vision model from {MODEL_PATH}...")
model = MllamaForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded successfully!")

def describe_erdes_label(diagnostic_class, subtype, anatomical_subclass):
    """
    Returns a clinical description of an ocular ultrasound clip based on
    the ERDES dataset labeling hierarchy.
    """
    diag = str(diagnostic_class).lower()
    sub = str(subtype).lower()
    anat = str(anatomical_subclass).upper()

    locations = {
        'TD': 'on the temporal side',
        'ND': 'on the nasal side',
        'BILATERAL': 'bilaterally (both sides)'
    }
    loc_desc = locations.get(anat, "")

    if diag == 'non_rd':
        if sub == 'normal':
            return ("A normal eye where the retina is a smooth, thin line closely "
                    "opposed to the globe with no mobile membranes.")
        elif sub == 'pvd':
            return ("Posterior Vitreous Detachment (PVD): A thin, highly mobile "
                    "membrane not tethered to the optic nerve.")

    elif diag == 'rd':
        if sub == 'macula_intact':
            return (f"Retinal Detachment {loc_desc} where the macula remains attached. ")
        elif sub == 'macula_detached':
            return (f"Retinal Detachment {loc_desc} that has extended to the macula. ")

    return "Unknown diagnostic combination."

def get_balanced_df(df, group_cols, random_state=42):
    group_sizes = df.groupby(group_cols).size()

    min_size = group_sizes.min()

    if min_size == 0:
        print("Warning: One or more groups have zero records. Cannot balance.")
        return pd.DataFrame(columns=df.columns)

    print(f"Balancing to {min_size} records per group for combinations of {group_cols}")

    balanced_list = []
    for name, group in df.groupby(group_cols):
        balanced_list.append(group.sample(n=min_size, random_state=random_state))
    balanced_df = pd.concat(balanced_list).reset_index(drop=True)

    return balanced_df

frames_to_extract = 3

def extract_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted_frames = []

    if num_frames > frame_count:
        num_frames = frame_count

    interval = frame_count // num_frames if num_frames > 0 else 0

    for i in range(num_frames):
        frame_idx = i * interval
        if frame_idx >= frame_count:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            extracted_frames.append(pil_image)
        else:
            print(f"Warning: Could not read frame {frame_idx}")

    cap.release()
    return extracted_frames


def generate_video_summary(record, extracted_frames):
    diagnosis = describe_erdes_label(record['diagnostic_class'], record['subtype'], record['anatomical_subclass'])

    all_frame_descriptions_llama = []

    if not diagnosis:
        print("No GT provided")
    else:
        print("GT provided:", diagnosis)

    if extracted_frames:
        print("\nGenerating descriptions for extracted frames using Llama 3.2 Vision...")
        
        for i, pil_image in enumerate(extracted_frames):
            try:
                image_prompt = PROMPT.format(DIAGNOSIS=diagnosis) if diagnosis else ASK_FOR_DIAGNOSIS_PROMPT
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": image_prompt}
                        ]
                    }
                ]
                
                input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(
                    pil_image,
                    input_text,
                    return_tensors="pt"
                ).to(model.device)
                
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=300)
                
                description = processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                all_frame_descriptions_llama.append(description)
            except Exception as e:
                print(f"Error processing frame {i}: {e}")

        if all_frame_descriptions_llama:
            print("\n--- Generating Video Summary (Llama) ---")
            combined_frame_texts = "\n".join(all_frame_descriptions_llama)

            image_prompt = SUMMARY_PROMPT.format(FRAME_DESCRIPTIONS=combined_frame_texts, DIAGNOSIS=diagnosis) if diagnosis else ASK_FOR_SUMMARY_PROMPT.format(FRAME_DESCRIPTIONS=combined_frame_texts)
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": image_prompt}
                        ]
                    }
                ]
                
                input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(
                    text=input_text,
                    return_tensors="pt"
                ).to(model.device)
                
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=500)
                
                video_summary_llama = processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                print("Based on the extracted frames, the video appears to show the following:\n")
                print(video_summary_llama)
                return video_summary_llama
            except Exception as e:
                print(f"Error generating summary with Llama: {e}")
                print("Fallback: Concatenating frame descriptions...")
                video_summary_llama = " ".join(all_frame_descriptions_llama)
                print("Based on the extracted frames, the video appears to show the following:\n")
                print(video_summary_llama)
                return None
        else:
            print("No frames were described by Llama.")
            return None
    else:
        print("Cannot describe video as no frames were extracted. Please ensure 'video_path' is correct and video file exists.")
        return None


def generate_video_summary_for_dataframe(balanced_split_desc):
    for row_id, record in tqdm(balanced_split_desc.iterrows(), total=balanced_split_desc.shape[0]):
        if type(record['summary']) is str:
            print(f"skip row {row_id}")
            continue
        video_path = os.path.join('/erdes', record['file_path'])
        if os.path.exists(video_path):
            extracted_frames = extract_frames(video_path, frames_to_extract)
        else:
            print(f"Video file '{video_path}' not found. Please upload a video or update the 'video_path' variable.")
            continue
        try:
            summary = generate_video_summary(record, extracted_frames)
            balanced_split_desc.loc[row_id, 'summary'] = summary
            balanced_split_desc.to_csv(SPLIT_DESC_CSV_SAVE_PATH)
        except Exception as e:
            print(f"Error in processing {row_id}")
            continue



def predict_video(balanced_split_desc):
    if 'predicted_retinal_detachment' not in balanced_split_desc.columns:
        balanced_split_desc['predicted_retinal_detachment'] = None
    if 'predicted_posterior_vitreous_detachment' not in balanced_split_desc.columns:
        balanced_split_desc['predicted_posterior_vitreous_detachment'] = None
    if 'predicted_macula_detached' not in balanced_split_desc.columns:
        balanced_split_desc['predicted_macula_detached'] = None
    if 'predicted_macula_intact' not in balanced_split_desc.columns:
        balanced_split_desc['predicted_macula_intact'] = None
    if 'predicted_summary' not in balanced_split_desc.columns:
        balanced_split_desc['predicted_summary'] = None

    for row_id, record in tqdm(balanced_split_desc.iterrows(), total=balanced_split_desc.shape[0]):
        if pd.notna(record['predicted_summary']) and pd.notna(record['predicted_macula_detached']) and pd.notna(record['predicted_macula_intact']):
            continue

        video_path = os.path.join('/erdes', record['file_path'])

        if os.path.exists(video_path):
            extracted_frames = extract_frames(video_path, frames_to_extract)
        else:
            print(f"Video file '{video_path}' not found. Skipping row {row_id}.")
            continue

        all_frame_descriptions_llama = []

        if extracted_frames:
            for i, pil_image in enumerate(extracted_frames):
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": ASK_FOR_DIAGNOSIS_PROMPT}
                            ]
                        }
                    ]
                    
                    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = processor(
                        pil_image,
                        input_text,
                        return_tensors="pt"
                    ).to(model.device)
                    
                    with torch.no_grad():
                        output = model.generate(**inputs, max_new_tokens=300)
                    
                    description = processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    all_frame_descriptions_llama.append(description)
                except Exception as e:
                    print(f"Error processing frame {i} for row {row_id}: {e}")
                    continue

            if all_frame_descriptions_llama:
                combined_frame_texts = "\n".join(all_frame_descriptions_llama)
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": ASK_FOR_SUMMARY_PROMPT.format(FRAME_DESCRIPTIONS=combined_frame_texts) + "\n\nIMPORTANT: Respond with valid JSON only, no additional text."}
                            ]
                        }
                    ]
                    
                    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = processor(
                        text=input_text,
                        return_tensors="pt"
                    ).to(model.device)
                    
                    with torch.no_grad():
                        output = model.generate(**inputs, max_new_tokens=500)
                    
                    response_text = processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                    
                    if response_text.startswith('```json'):
                        response_text = response_text[7:]
                    if response_text.startswith('```'):
                        response_text = response_text[3:]
                    if response_text.endswith('```'):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()
                    
                    json_summary = json.loads(response_text)

                    balanced_split_desc.loc[row_id, 'predicted_retinal_detachment'] = json_summary.get('Retinal Detachment')
                    balanced_split_desc.loc[row_id, 'predicted_posterior_vitreous_detachment'] = json_summary.get('Posterior Vitreous Detachment')
                    balanced_split_desc.loc[row_id, 'predicted_macula_detached'] = json_summary.get('Macula Detached')
                    balanced_split_desc.loc[row_id, 'predicted_macula_intact'] = json_summary.get('Macula Intact')
                    balanced_split_desc.loc[row_id, 'predicted_summary'] = json_summary.get('Summary')

                    balanced_split_desc.to_csv(SPLIT_DESC_CSV_SAVE_PATH, index=False)

                    print(f"Row {row_id}: RD: {json_summary.get('Retinal Detachment')}, PVD: {json_summary.get('Posterior Vitreous Detachment')}, Macula Detached: {json_summary.get('Macula Detached')}, Macula Intact: {json_summary.get('Macula Intact')}, Summary: {json_summary.get('Summary', '')[:75]}...")

                except Exception as e:
                    print(f"Error generating summary for row {row_id} with Llama: {e}")
                    balanced_split_desc.loc[row_id, 'predicted_summary'] = "Error: " + str(e)
                    balanced_split_desc.to_csv(SPLIT_DESC_CSV_SAVE_PATH, index=False)
                    continue
            else:
                print(f"No frames were described by Llama for row {row_id}.")
                balanced_split_desc.loc[row_id, 'predicted_summary'] = "No frames described."
                balanced_split_desc.to_csv(SPLIT_DESC_CSV_SAVE_PATH, index=False)
        else:
            print(f"Cannot describe video as no frames were extracted for row {row_id}.")
            balanced_split_desc.loc[row_id, 'predicted_summary'] = "No frames extracted."
            balanced_split_desc.to_csv(SPLIT_DESC_CSV_SAVE_PATH, index=False)

    print("Processing complete. The updated dataframe with predictions has been saved to CSV.")


if __name__ == "__main__":
    balanced_split_desc = pd.read_csv(SPLIT_DESC_CSV_PATH)
    if os.path.exists(SPLIT_DESC_CSV_SAVE_PATH):
        balanced_split_desc = pd.read_csv(SPLIT_DESC_CSV_SAVE_PATH)
        print("loading existing preidiction")
    predict_video(balanced_split_desc)
