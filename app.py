# app.py
import gradio as gr
import joblib
import numpy as np

# Load trained models
clf_dosha = joblib.load("dosha_model.joblib")
reg_days = joblib.load("days_model.joblib")

# Therapy mapping for each Dosha (all 5 Panchakarma therapies)
therapy_map = {
    "Vata": [
        "Vamana (Therapeutic Emesis)",
        "Virechana (Purgation)",
        "Basti (Medicated Enema)",
        "Nasya (Nasal Therapy)",
        "Raktamokshana (Bloodletting)"
    ],
    "Pitta": [
        "Vamana (Therapeutic Emesis)",
        "Virechana (Purgation)",
        "Basti (Medicated Enema)",
        "Nasya (Nasal Therapy)",
        "Raktamokshana (Bloodletting)"
    ],
    "Kapha": [
        "Vamana (Therapeutic Emesis)",
        "Virechana (Purgation)",
        "Basti (Medicated Enema)",
        "Nasya (Nasal Therapy)",
        "Raktamokshana (Bloodletting)"
    ]
}

# Prediction function
def predict_plan(age, weight, sleep, stress, appetite, energy, digestion, gender, medical_history):
    # Encode categorical
    map_dict = {"Low": 0, "Medium": 1, "High": 2, "Weak": 0, "Normal": 1, "Strong": 2, "Male": 0, "Female": 1, "Other": 2}
    gender_encoded = map_dict[gender]
    medical_history_encoded = 1 if medical_history.strip() else 0
    features = np.array([[age, weight, sleep, stress,
                          map_dict[appetite], map_dict[energy], map_dict[digestion],
                          gender_encoded, medical_history_encoded]])
    # Predict dosha
    dosha = clf_dosha.predict(features)[0]
    # Predict days
    predicted_days = int(reg_days.predict(features)[0])
    # Recommend first therapy for dosha
    therapy_choice = therapy_map.get(dosha, ["General Wellness"])[0]
    return f"""
    🌀 **Predicted Dosha:** {dosha}  
    💆 **Recommended Therapy:** {therapy_choice}  
    ⏳ **Estimated Duration:** {predicted_days} days
    """

# Build UI
def get_therapy_choices(dosha="Vata"):
    return therapy_map.get(dosha, ["General Wellness"])

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image("logo.png", width=40, show_label=False)
        with gr.Column(scale=6):
            gr.Markdown("# PranaSuddhi - Pure Life. Pure You.")

    with gr.Row():
        with gr.Column():
            age = gr.Number(label="Age")
            weight = gr.Number(label="Weight (kg)")
            sleep = gr.Slider(3, 12, step=1, label="Sleep Hours")
            stress = gr.Slider(1, 5, step=1, label="Stress Level (1=Low, 5=High)")
            appetite = gr.Radio(["Low", "Medium", "High"], label="Appetite")
            energy = gr.Radio(["Low", "Medium", "High"], label="Energy Level")
            digestion = gr.Radio(["Weak", "Normal", "Strong"], label="Digestion Strength")
            gender = gr.Radio(["Male", "Female", "Other"], label="Gender")
            medical_history = gr.Textbox(label="Medical History (chronic diseases, allergies)")
            submit = gr.Button("Predict Plan")

        with gr.Column():
            output = gr.Markdown()

    submit.click(fn=predict_plan,
                 inputs=[age, weight, sleep, stress, appetite, energy, digestion, gender, medical_history],
                 outputs=output)

# Launch app
if __name__ == "__main__":
    demo.launch(share=True)
