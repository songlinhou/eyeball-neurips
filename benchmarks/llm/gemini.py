import google.generativeai as genai
import base64
import pandas as pd
import cv2
from PIL import Image
import os
import io
from tqdm.auto import tqdm
import json
from prompts import ASK_FOR_DIAGNOSIS_PROMPT, PROMPT, SUMMARY_PROMPT, ASK_FOR_SUMMARY_PROMPT
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# PROJECT_DIR = "drive/MyDrive/EyeballProject"
SPLIT_DESC_CSV_PATH = '../input/balanced_split_desc.csv'
SPLIT_DESC_CSV_SAVE_PATH = '../output/gemini_prediction.csv'


# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

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
    extracted_frames_data = []

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
            _, buffer = cv2.imencode('.jpg', frame)
            base64_frame = base64.b64encode(buffer).decode('utf-8')
            extracted_frames_data.append(base64_frame)
        else:
            print(f"Warning: Could not read frame {frame_idx}")

    cap.release()
    return extracted_frames_data


def generate_video_summary(record, extracted_frame_data):
    diagnosis = describe_erdes_label(record['diagnostic_class'], record['subtype'], record['anatomical_subclass'])

    all_frame_descriptions_gemini = []

    if not diagnosis:
        print("No GT provided")
    else:
        print("GT provided:", diagnosis)

    if extracted_frame_data:
        print("\nGenerating descriptions for extracted frames using Gemini...")
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        for i, base64_image in enumerate(extracted_frame_data):
            try:
                image_bytes = base64.b64decode(base64_image)
                image = Image.open(io.BytesIO(image_bytes))
                
                image_prompt = PROMPT.format(DIAGNOSIS=diagnosis) if diagnosis else ASK_FOR_DIAGNOSIS_PROMPT
                
                response = model.generate_content([image_prompt, image])
                description = response.text
                all_frame_descriptions_gemini.append(description)
            except Exception as e:
                print(f"Error processing frame {i}: {e}")

        if all_frame_descriptions_gemini:
            print("\n--- Generating Video Summary (Gemini) ---")
            combined_frame_texts = "\n".join(all_frame_descriptions_gemini)

            image_prompt = SUMMARY_PROMPT.format(FRAME_DESCRIPTIONS=combined_frame_texts, DIAGNOSIS=diagnosis) if diagnosis else ASK_FOR_SUMMARY_PROMPT.format(FRAME_DESCRIPTIONS=combined_frame_texts)
            try:
                summary_response = model.generate_content(image_prompt)
                video_summary_gemini = summary_response.text
                print("Based on the extracted frames, the video appears to show the following:\n")
                print(video_summary_gemini)
                return video_summary_gemini
            except Exception as e:
                print(f"Error generating summary with Gemini: {e}")
                print("Fallback: Concatenating frame descriptions...")
                video_summary_gemini = " ".join(all_frame_descriptions_gemini)
                print("Based on the extracted frames, the video appears to show the following:\n")
                print(video_summary_gemini)
                return None
        else:
            print("No frames were described by Gemini.")
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
            extracted_frame_data = extract_frames(video_path, frames_to_extract)
        else:
            print(f"Video file '{video_path}' not found. Please upload a video or update the 'video_path' variable.")
            continue
        try:
            summary = generate_video_summary(record, extracted_frame_data)
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

    model = genai.GenerativeModel('gemini-2.5-flash')

    for row_id, record in tqdm(balanced_split_desc.iterrows(), total=balanced_split_desc.shape[0]):
        if pd.notna(record['predicted_summary']) and pd.notna(record['predicted_macula_detached']) and pd.notna(record['predicted_macula_intact']):
            continue

        video_path = os.path.join('/erdes', record['file_path'])

        if os.path.exists(video_path):
            extracted_frame_data = extract_frames(video_path, frames_to_extract)
        else:
            print(f"Video file '{video_path}' not found. Skipping row {row_id}.")
            continue

        all_frame_descriptions_gemini = []

        if extracted_frame_data:
            for i, base64_image in enumerate(extracted_frame_data):
                try:
                    image_bytes = base64.b64decode(base64_image)
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    response = model.generate_content([ASK_FOR_DIAGNOSIS_PROMPT, image])
                    description = response.text
                    all_frame_descriptions_gemini.append(description)
                except Exception as e:
                    print(f"Error processing frame {i} for row {row_id}: {e}")
                    continue

            if all_frame_descriptions_gemini:
                combined_frame_texts = "\n".join(all_frame_descriptions_gemini)
                try:
                    generation_config = {
                        "response_mime_type": "application/json"
                    }
                    
                    summary_response = model.generate_content(
                        ASK_FOR_SUMMARY_PROMPT.format(FRAME_DESCRIPTIONS=combined_frame_texts),
                        generation_config=generation_config
                    )
                    json_summary = json.loads(summary_response.text)

                    balanced_split_desc.loc[row_id, 'predicted_retinal_detachment'] = json_summary.get('Retinal Detachment')
                    balanced_split_desc.loc[row_id, 'predicted_posterior_vitreous_detachment'] = json_summary.get('Posterior Vitreous Detachment')
                    balanced_split_desc.loc[row_id, 'predicted_macula_detached'] = json_summary.get('Macula Detached')
                    balanced_split_desc.loc[row_id, 'predicted_macula_intact'] = json_summary.get('Macula Intact')
                    balanced_split_desc.loc[row_id, 'predicted_summary'] = json_summary.get('Summary')

                    balanced_split_desc.to_csv(SPLIT_DESC_CSV_SAVE_PATH, index=False)

                    print(f"Row {row_id}: RD: {json_summary.get('Retinal Detachment')}, PVD: {json_summary.get('Posterior Vitreous Detachment')}, Macula Detached: {json_summary.get('Macula Detached')}, Macula Intact: {json_summary.get('Macula Intact')}, Summary: {json_summary.get('Summary', '')[:75]}...")

                except Exception as e:
                    print(f"Error generating summary for row {row_id} with Gemini: {e}")
                    balanced_split_desc.loc[row_id, 'predicted_summary'] = "Error: " + str(e)
                    balanced_split_desc.to_csv(SPLIT_DESC_CSV_SAVE_PATH, index=False)
                    continue
            else:
                print(f"No frames were described by Gemini for row {row_id}.")
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
