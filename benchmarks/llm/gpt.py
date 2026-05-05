from openai import OpenAI
import base64
import pandas as pd
import cv2
from PIL import Image
import os
import io # Added for displaying base64 images
from tqdm.auto import tqdm
import json
from prompts import ASK_FOR_DIAGNOSIS_PROMPT, PROMPT, SUMMARY_PROMPT, ASK_FOR_SUMMARY_PROMPT
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# PROJECT_DIR = "drive/MyDrive/EyeballProject"
SPLIT_DESC_CSV_PATH = '../input/balanced_split_desc.csv'
SPLIT_DESC_CSV_SAVE_PATH = '../output/gpt_prediction.csv'
ERDES = '/erdes'
if not os.path.exists(ERDES):
    ERDES = '/content/eyeball-neurips/erdes'

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def describe_erdes_label(diagnostic_class, subtype, anatomical_subclass):
    """
    Returns a clinical description of an ocular ultrasound clip based on
    the ERDES dataset labeling hierarchy.
    """
    # Normalize inputs
    diag = str(diagnostic_class).lower()
    sub = str(subtype).lower()
    anat = str(anatomical_subclass).upper()

    # Mapping for anatomical locations
    locations = {
        'TD': 'on the temporal side',
        'ND': 'on the nasal side',
        'BILATERAL': 'bilaterally (both sides)'
    }
    loc_desc = locations.get(anat, "")

    # Logic based on ERDES classification criteria
    if diag == 'non_rd':
        if sub == 'normal':
            return ("A normal eye where the retina is a smooth, thin line closely "
                    "opposed to the globe with no mobile membranes.")
        elif sub == 'pvd':
            return ("Posterior Vitreous Detachment (PVD): A thin, highly mobile "
                    "membrane not tethered to the optic nerve.")

    elif diag == 'rd':
        if sub == 'macula_intact':
            # return (f"Retinal Detachment {loc_desc} where the macula remains attached. "
            #         "This is a surgical emergency requiring intervention within 24 hours.")
            return (f"Retinal Detachment {loc_desc} where the macula remains attached. ")
        elif sub == 'macula_detached':
            # return (f"Retinal Detachment {loc_desc} that has extended to the macula, "
            #         "which typically indicates a poorer visual prognosis.")
            return (f"Retinal Detachment {loc_desc} that has extended to the macula. ")

    return "Unknown diagnostic combination."

def get_balanced_df(df, group_cols, random_state=42):
    # Group by the specified columns and get the size of each group
    group_sizes = df.groupby(group_cols).size()

    # Determine the minimum size among all groups
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

frames_to_extract = 5 # Number of frames to extract for description (adjusted to match previous output)

def extract_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted_frames_data = [] # Changed to store base64 encoded image data

    if num_frames > frame_count:
        num_frames = frame_count

    # Calculate interval to extract frames evenly, ensuring num_frames is not zero
    interval = frame_count // num_frames if num_frames > 0 else 0

    for i in range(num_frames):
        frame_idx = i * interval
        if frame_idx >= frame_count:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            _, buffer = cv2.imencode('.jpg', frame) # Encode frame to JPG bytes
            base64_frame = base64.b64encode(buffer).decode('utf-8') # Convert bytes to base64 string
            extracted_frames_data.append(base64_frame)
        else:
            print(f"Warning: Could not read frame {frame_idx}")

    cap.release()
    # print(f"Extracted {len(extracted_frames_data)} frames into memory.") # Updated print statement
    return extracted_frames_data # Return list of base64 strings


def generate_video_summary(record, extracted_frame_data):
    diagnosis = describe_erdes_label(record['diagnostic_class'], record['subtype'], record['anatomical_subclass'])
    # diagnosis = None

    all_frame_descriptions_openai = []

    if not diagnosis:
        print("No GT provided")
    else:
        print("GT provided:", diagnosis)

    if extracted_frame_data: # Changed variable name from extracted_frame_files
        print("\nGenerating descriptions for extracted frames using OpenAI...")
        for i, base64_image in enumerate(extracted_frame_data): # Iterate directly over base64 strings
            try:
                # base64_image is already encoded
                image_prompt = PROMPT.format(DIAGNOSIS=diagnosis) if diagnosis else ASK_FOR_DIAGNOSIS_PROMPT
                response = client.chat.completions.create(
                    model="gpt-5.4", # You can also try "gpt-4-vision-preview" if you prefer.
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": image_prompt,},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}
                                },
                            ],
                        }
                    ],
                    max_tokens=300,
                )
                description = response.choices[0].message.content
                # print(f"\nDescription for frame_{i:04d}.jpg:\n{description}") # Adjusted print to show frame index
                all_frame_descriptions_openai.append(description)
            except Exception as e:
                print(f"Error processing frame {i}: {e}") # Adjusted error message

        if all_frame_descriptions_openai:
            print("\n--- Generating Video Summary (OpenAI) ---")
            # Use another LLM call to synthesize the frame descriptions into a video summary
            combined_frame_texts = "\n".join(all_frame_descriptions_openai)

            image_prompt = SUMMARY_PROMPT.format(FRAME_DESCRIPTIONS=combined_frame_texts, DIAGNOSIS=diagnosis) if diagnosis else ASK_FOR_SUMMARY_PROMPT.format(FRAME_DESCRIPTIONS=combined_frame_texts)
            try:
                summary_response = client.chat.completions.create(
                    model="gpt-4o", # Using the same model for consistency
                    messages=[
                        {
                            "role": "user",
                            "content": image_prompt
                        }
                    ],
                    max_tokens=500, # Increased max_tokens for a potentially longer summary
                )
                video_summary_openai = summary_response.choices[0].message.content
                print("Based on the extracted frames, the video appears to show the following:\n")
                print(video_summary_openai)
                return video_summary_openai
            except Exception as e:
                print(f"Error generating summary with OpenAI: {e}")
                print("Fallback: Concatenating frame descriptions...")
                video_summary_openai = " ".join(all_frame_descriptions_openai)
                print("Based on the extracted frames, the video appears to show the following:\n")
                print(video_summary_openai)
                return None
        else:
            print("No frames were described by OpenAI.")
            return None
    else:
        print("Cannot describe video as no frames were extracted. Please ensure 'video_path' is correct and video file exists.")
        return None


def generate_video_summary_for_dataframe(balanced_split_desc):
    for row_id, record in tqdm(balanced_split_desc.iterrows(), total=balanced_split_desc.shape[0]):
        if type(record['summary']) is str:
            print(f"skip row {row_id}")
            continue
        video_path = os.path.join(ERDES, record['file_path'])  # <<< IMPORTANT: Replace with your video file path
        if os.path.exists(video_path):
            extracted_frame_data = extract_frames(video_path, frames_to_extract) # Updated function call and variable name
            # if extracted_frame_data:
            #     print("\nDisplaying the first extracted frame:")
            #     # Decode the first base64 string to display it
            #     decoded_image_bytes = base64.b64decode(extracted_frame_data[0])
            #     display(Image.open(io.BytesIO(decoded_image_bytes)))
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
        # Initialize new columns for predictions
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
        # Skip if already processed (check 'predicted_summary' and new macula columns)
        if pd.notna(record['predicted_summary']) and pd.notna(record['predicted_macula_detached']) and pd.notna(record['predicted_macula_intact']):
            # print(f"Skipping row {row_id} as it's already processed.")
            continue

        video_path = os.path.join(ERDES, record['file_path'])

        if os.path.exists(video_path):
            extracted_frame_data = extract_frames(video_path, frames_to_extract)
        else:
            print(f"Video file '{video_path}' not found. Skipping row {row_id}.")
            continue

        all_frame_descriptions_openai = []

        if extracted_frame_data:
            for i, base64_image in enumerate(extracted_frame_data):
                try:
                    response = client.chat.completions.create(
                        model="gpt-5.4-mini",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": ASK_FOR_DIAGNOSIS_PROMPT},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}
                                    },
                                ],
                            }
                        ],
                        max_completion_tokens=300,
                    )
                    description = response.choices[0].message.content
                    all_frame_descriptions_openai.append(description)
                except Exception as e:
                    print(f"Error processing frame {i} for row {row_id}: {e}")
                    continue

            if all_frame_descriptions_openai:
                combined_frame_texts = "\n".join(all_frame_descriptions_openai)
                try:
                    summary_response = client.chat.completions.create(
                        model="gpt-5.4-mini",
                        messages=[
                            {
                                "role": "user",
                                "content": ASK_FOR_SUMMARY_PROMPT.format(FRAME_DESCRIPTIONS=combined_frame_texts)
                            }
                        ],
                        max_completion_tokens=500,
                        response_format={ "type": "json_object" }
                    )
                    json_summary = json.loads(summary_response.choices[0].message.content)

                    balanced_split_desc.loc[row_id, 'predicted_retinal_detachment'] = json_summary.get('Retinal Detachment')
                    balanced_split_desc.loc[row_id, 'predicted_posterior_vitreous_detachment'] = json_summary.get('Posterior Vitreous Detachment')
                    balanced_split_desc.loc[row_id, 'predicted_macula_detached'] = json_summary.get('Macula Detached')
                    balanced_split_desc.loc[row_id, 'predicted_macula_intact'] = json_summary.get('Macula Intact')
                    balanced_split_desc.loc[row_id, 'predicted_summary'] = json_summary.get('Summary')

                    balanced_split_desc.to_csv(SPLIT_DESC_CSV_SAVE_PATH, index=False)

                    # Print the prediction for the current row
                    print(f"Row {row_id}: RD: {json_summary.get('Retinal Detachment')}, PVD: {json_summary.get('Posterior Vitreous Detachment')}, Macula Detached: {json_summary.get('Macula Detached')}, Macula Intact: {json_summary.get('Macula Intact')}, Summary: {json_summary.get('Summary', '')[:75]}...")

                except Exception as e:
                    print(f"Error generating summary for row {row_id} with OpenAI: {e}")
                    balanced_split_desc.loc[row_id, 'predicted_summary'] = "Error: " + str(e)
                    balanced_split_desc.to_csv(SPLIT_DESC_CSV_SAVE_PATH, index=False)
                    continue
            else:
                print(f"No frames were described by OpenAI for row {row_id}.")
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
