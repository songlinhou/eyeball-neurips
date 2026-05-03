ASK_FOR_DIAGNOSIS_PROMPT = """
You are a doctor in examing the ocular ultrasound videos.
This is one frame of an ocular utrasound video.
Analyze this video and give diagnosis from your observations.
Explain your diagnosis with what you find when analyzing the frame.

The diagnosis should at least include:
1. Retinal Detachment or not?
2. If it is not Retinal Detachment, is it Posterior Vitreous Detachment?
3. Is the macula detached or intact?

If you cannot determine, just mention you do not know.
"""

PROMPT = """
You are a doctor in examing the ocular ultrasound videos.
This is one frame of an ocular utrasound video.
Analyze this video and describe why the diagnosis is made from your observations.
Do not mention justifying or confirming the provided diagnosis since the audience has no knowledge of the mentioned diagnosis.
Only answer based on given information. Also answer in one paragraph.

This is the diagnosis:
{DIAGNOSIS}
"""

SUMMARY_PROMPT = """
You are an expert in summarizing medical observations.
Below are descriptions of individual frames from an ocular ultrasound video.
Please synthesize these descriptions into a concise and coherent video summary, highlighting key medical findings and conclusions about the patient's condition as observed across the entire video.

Main Diagnosis:
{DIAGNOSIS}

Frame Descriptions:
{FRAME_DESCRIPTIONS}
"""

ASK_FOR_SUMMARY_PROMPT = """
You are an expert in summarizing medical observations.
Below are descriptions of individual frames from an ocular ultrasound video.
Please synthesize these descriptions into a concise and coherent video summary, highlighting key medical findings and conclusions about the patient's condition as observed across the entire video.

Frame Descriptions:
{FRAME_DESCRIPTIONS}


Respond in the following JSON format:
{{
   "Retinal Detachment": <bool>,
   "Posterior Vitreous Detachment": <bool>,
   "Macula Detached": <bool>,
   "Macula Intact": <bool>,
   "Summary": <str>
}}
"""
