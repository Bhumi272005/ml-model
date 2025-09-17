# progress_app.py
import gradio as gr
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load progress model
reg_progress = joblib.load("progress_model.joblib")

# Prediction function with graph
def predict_progress(sessions_completed, severity, adherence, symptom_score):
    map_severity = {"Mild": 1, "Moderate": 2, "Severe": 3}
    map_adherence = {"Low": 0, "Medium": 1, "High": 2}

    features = np.array([[sessions_completed, 
                          map_severity[severity], 
                          map_adherence[adherence], 
                          symptom_score]])
    
    progress = int(reg_progress.predict(features)[0])

    # Interpret progress
    if progress < 40:
        status = "⚠️ Needs Attention"
    elif progress < 70:
        status = "📈 Improving"
    else:
        status = "🌟 Excellent Recovery"

    # Return only text result
    return f"""
    ✅ **Predicted Progress Score:** {progress}%  
    📊 **Recovery Status:** {status}
    """

# Gradio UI
demo = gr.Interface(
    fn=predict_progress,
    inputs=[
        gr.Number(label="Therapy Sessions Completed"),
        gr.Radio(["Mild", "Moderate", "Severe"], label="Initial Dosha Severity"),
        gr.Radio(["Low", "Medium", "High"], label="Lifestyle Adherence"),
        gr.Slider(0, 100, step=5, label="Symptom Improvement Score (0-100)")
    ],
    outputs=gr.Markdown(),
    title="📈 Patient Progress Predictor",
    description="Track how well the patient is improving during therapy with prediction + graph."
)

if __name__ == "__main__":
    demo.launch()
