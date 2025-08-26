import json
import requests
from typing import Dict

URL = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}

def emotion_detector(text_to_analyze: str) -> Dict[str, float | str]:
    """
    Calls Watson NLP EmotionPredict and returns:
    {
      'anger': float,
      'disgust': float,
      'fear': float,
      'joy': float,
      'sadness': float,
      'dominant_emotion': str
    }
    or all None on blank input or 400 error.
    """
    if not text_to_analyze.strip():  # Handle blank/empty input
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }

    payload = {"raw_document": {"text": text_to_analyze}}
    resp = requests.post(URL, headers=HEADERS, json=payload, timeout=30)

    if resp.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }

    resp.raise_for_status()  # Raise for other errors

    data = json.loads(resp.text)

    preds = data.get("emotionPredictions", [])
    if not preds:
        return {
            "anger": 0.0,
            "disgust": 0.0,
            "fear": 0.0,
            "joy": 0.0,
            "sadness": 0.0,
            "dominant_emotion": None,
        }

    # Prefer overall document prediction (target == ""), otherwise use first element
    doc_pred = next((p for p in preds if p.get("target", "") == ""), preds[0])
    emo = doc_pred.get("emotion", {})

    result = {
        "anger": float(emo.get("anger", 0.0)),
        "disgust": float(emo.get("disgust", 0.0)),
        "fear": float(emo.get("fear", 0.0)),
        "joy": float(emo.get("joy", 0.0)),
        "sadness": float(emo.get("sadness", 0.0)),
    }
    result["dominant_emotion"] = max(result, key=result.get)
    return result

if __name__ == "__main__":
    sample = "I am so happy I am doing this."
    print(json.dumps(emotion_detector(sample), indent=2))
